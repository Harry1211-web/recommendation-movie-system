"""
app.py — Movie Recommendation System (Optimized)
Streamlit web app: Hybrid SVD + ItemCF + IMDb Boost + Content-Based cold-start.

Optimizations applied:
  1. @st.cache_data on recommend pipeline → skip recompute for same user/model
  2. Batch SVD prediction (matrix ops) instead of 3000 individual calls
  3. Vectorized ItemCF (precomputed per-user, full cosine similarity matrix)
  4. Smart candidate selection (popular-first + random diversity)
  5. Cached helper functions (is_cold_start, user history)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity as cossim

warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Styling ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%); }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1f 0%, #1a1a2e 100%);
        border-right: 1px solid #e9456030;
    }

    .movie-card {
        background: linear-gradient(135deg, #1e1e35 0%, #252540 100%);
        border: 1px solid #e9456025;
        border-radius: 16px;
        padding: 18px 16px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #e94560, #533483, #0f3460);
    }
    .movie-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 30px rgba(233, 69, 96, 0.25);
        border-color: #e9456055;
    }

    .movie-title  { font-size: 1.05rem; font-weight: 700; color: #ffffff; margin-bottom: 6px; line-height: 1.3; }
    .movie-genres { font-size: 0.78rem; color: #8888aa; margin-bottom: 10px; }
    .movie-score-row { display: flex; justify-content: space-between; align-items: center; margin-top: 10px; }
    .badge-pred  { background: linear-gradient(135deg, #e94560, #c0392b); color: white; border-radius: 20px; padding: 4px 12px; font-size: 0.82rem; font-weight: 600; }
    .badge-imdb  { background: linear-gradient(135deg, #f39c12, #e2b04a); color: #1a1a2e; border-radius: 20px; padding: 4px 10px; font-size: 0.80rem; font-weight: 700; }
    .badge-model { background: rgba(83, 52, 131, 0.5); color: #c39bd3; border-radius: 12px; padding: 2px 8px; font-size: 0.72rem; }

    .metric-card  { background: linear-gradient(135deg, #1e1e35, #252540); border: 1px solid #ffffff10; border-radius: 14px; padding: 18px; text-align: center; transition: all 0.25s ease; }
    .metric-card:hover { border-color: #e9456040; }
    .metric-title { font-size: 0.78rem; color: #8888aa; margin-bottom: 6px; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-good  { color: #2ecc71; }
    .metric-mid   { color: #e2b04a; }
    .metric-bad   { color: #e94560; }

    .section-header { font-size: 1.4rem; font-weight: 700; color: #ffffff; border-left: 4px solid #e94560; padding-left: 12px; margin: 24px 0 16px 0; }
    .history-item   { background: #1a1a2e; border: 1px solid #ffffff10; border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.88rem; color: #ccccdd; }

    hr { border: none; border-top: 1px solid #ffffff10; margin: 20px 0; }
    .stRadio > label { color: #ccccdd !important; }
    div[data-testid="stRadio"] label { color: #ccccdd !important; }
</style>
""", unsafe_allow_html=True)


# ─── Load Models (cached across sessions) ──────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner=False)
def load_models():
    meta        = joblib.load(MODELS_DIR / "model_metadata.pkl")
    svd_model   = joblib.load(MODELS_DIR / "svd_model.pkl")
    itemcf_pkg  = joblib.load(MODELS_DIR / "itemcf_model.pkl")
    content_pkg = joblib.load(MODELS_DIR / "content_model.pkl")
    df_train    = pd.read_csv(MODELS_DIR / "df_train.csv")
    df_movies   = pd.read_csv(MODELS_DIR / "movies_metadata_for_testing.csv")

    # ── Precompute item neighbors một lần ──────────────────────────────────
    knn_item = itemcf_pkg["knn_item"]
    R_item   = itemcf_pkg["R_item"]
    n_neighbors = meta.get("n_neighbors", 50)
    dist_pre, idx_pre = knn_item.kneighbors(R_item, n_neighbors=n_neighbors + 1)
    itemcf_pkg["neighbor_idx"] = idx_pre[:, 1:]       # (n_items, N)
    itemcf_pkg["neighbor_sim"] = 1 - dist_pre[:, 1:]  # (n_items, N)
    del dist_pre, idx_pre

    return meta, svd_model, itemcf_pkg, content_pkg, df_train, df_movies

with st.spinner("Loading models..."):
    meta, svd_model, itemcf_pkg, content_pkg, df_train, df_movies = load_models()

# ─── Unpack metadata ───────────────────────────────────────────────────────────
user_to_idx   = meta["user_to_idx"]
movie_to_idx  = meta["movie_to_idx"]
BEST_ALPHA    = meta["best_alpha"]
BEST_BETA     = meta["best_beta"]
imdb_norm_map = meta["imdb_norm_map"]
R_MIN         = meta["rating_min"]
R_MAX         = meta["rating_max"]
COLD_THR      = meta.get("cold_start_threshold", 5)
N_NEIGHBORS   = meta.get("n_neighbors", 50)
RMSE          = meta.get("rmse_scores", {})
MAE           = meta.get("mae_scores", {})
P10           = meta.get("precision_at_10", {})
R10           = meta.get("recall_at_10", {})
ACTIVITY_CAP  = meta.get("activity_cap", 200)
ALPHA_SWING   = meta.get("alpha_swing", 0.3)

# ─── Unpack model objects ──────────────────────────────────────────────────────
knn_item      = itemcf_pkg["knn_item"]
R_mat         = itemcf_pkg["R"]
R_item        = itemcf_pkg["R_item"]
user_profiles = content_pkg["user_profile"]
movie_vec     = content_pkg["movie_vec"]

movie_ids_all = list(movie_to_idx.keys())

df_movies["genres_str"] = df_movies["genres"].apply(
    lambda g: g if isinstance(g, str) else "|".join(g) if isinstance(g, list) else ""
)

# ─── Precompute popular movie order (once at startup) ─────────────────────────
# Used for smart candidate selection: popular-first + random diversity
popular_movies_ordered = (
    df_train.groupby("movieId").size()
    .sort_values(ascending=False)
    .index.tolist()
)
popular_movies_set = set(popular_movies_ordered)

# Precompute per-user rating counts (avoid repeated df scans)
user_rating_counts = df_train.groupby("userId").size().to_dict()


# ══════════════════════════════════════════════════════════════════════════════
#  OPTIMIZED PREDICTION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

# ─── OPT 1: Batch SVD (matrix ops, no per-item loop) ─────────────────────────
@st.cache_data(show_spinner=False)
def predict_svd_batch(uid, candidate_tuple):
    ts = svd_model.trainset
    gm = ts.global_mean
    try:
        inner_uid = ts.to_inner_uid(uid)
    except ValueError:
        return {mid: float(gm) for mid in candidate_tuple}

    pu = svd_model.pu[inner_uid]   # (n_factors,)
    bu = svd_model.bu[inner_uid]

    # Map candidates → inner ids (bỏ unknown)
    valid, inner_iids = [], []
    for mid in candidate_tuple:
        try:
            inner_iids.append(ts.to_inner_iid(mid))
            valid.append(mid)
        except ValueError:
            pass

    if not valid:
        return {mid: float(gm) for mid in candidate_tuple}

    # Vectorized: tất cả items cùng lúc
    qi_mat = svd_model.qi[inner_iids]           # (n_valid, n_factors)
    bi_arr = svd_model.bi[inner_iids]           # (n_valid,)
    scores = np.clip(gm + bu + bi_arr + qi_mat.dot(pu), R_MIN, R_MAX)

    result = {mid: float(gm) for mid in candidate_tuple}  # default unknown
    for mid, score in zip(valid, scores):
        result[mid] = float(score)
    return result

# ─── OPT 2: Vectorized ItemCF (full matrix, computed once per user) ───────────
# Unpack thêm sau load_models()
item_neighbor_idx = itemcf_pkg["neighbor_idx"]
item_neighbor_sim = itemcf_pkg["neighbor_sim"]

@st.cache_data(show_spinner=False)
def predict_itemcf_batch(uid, candidate_tuple):
    if uid not in user_to_idx:
        return {}

    uidx     = user_to_idx[uid]
    user_row = R_mat[uidx].toarray().flatten()   # (n_items,)

    mid_list  = [m for m in candidate_tuple if m in movie_to_idx]
    if not mid_list:
        return {}

    midxs         = np.array([movie_to_idx[m] for m in mid_list])
    neighbor_idxs = item_neighbor_idx[midxs]      # (n_cands, N) — lookup O(1)
    neighbor_sims = item_neighbor_sim[midxs]      # (n_cands, N)
    neighbor_rats = user_row[neighbor_idxs]       # (n_cands, N)

    rated_mask   = neighbor_rats > 0
    weighted_sum = (neighbor_sims * neighbor_rats * rated_mask).sum(axis=1)
    sim_sum      = (neighbor_sims * rated_mask).sum(axis=1)
    scores       = np.where(sim_sum > 0, weighted_sum / (sim_sum + 1e-9), np.nan)

    return {m: float(s) for m, s in zip(mid_list, scores)}

# ─── Content-Based (single, fast enough) ──────────────────────────────────────
def predict_content_single(uid, mid):
    if uid not in user_profiles or mid not in movie_vec:
        return np.nan
    try:
        sim = cossim(user_profiles[uid].reshape(1, -1),
                     movie_vec[mid].toarray())[0][0]
        return float(R_MIN + sim * (R_MAX - R_MIN))
    except Exception:
        return np.nan


@st.cache_data(show_spinner=False)
def predict_content_batch(uid, candidate_tuple):
    """Vectorized content-based: score all candidates at once."""
    if uid not in user_profiles:
        return {}
    uprof = user_profiles[uid].reshape(1, -1)  # (1, n_features)

    valid_mids = [m for m in candidate_tuple if m in movie_vec]
    if not valid_mids:
        return {}

    # Stack movie vectors
    import scipy.sparse as sp
    movie_matrix = sp.vstack([movie_vec[m] for m in valid_mids])  # (n_valid, n_features)
    sims = cossim(uprof, movie_matrix)[0]                          # (n_valid,)
    scores = R_MIN + sims * (R_MAX - R_MIN)

    return {m: float(s) for m, s in zip(valid_mids, scores)}


# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_user_rating_count(uid):
    return int(user_rating_counts.get(uid, 0))

def is_cold_start(uid):
    return get_user_rating_count(uid) < COLD_THR

def adaptive_alpha_fn(uid):
    n              = get_user_rating_count(uid)
    activity_ratio = min(1.0, n / ACTIVITY_CAP)
    return float(np.clip(BEST_ALPHA + ALPHA_SWING * (1 - 2 * activity_ratio), 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
#  CORE RECOMMEND FUNCTION — fully cached per (uid, model, params)
# ══════════════════════════════════════════════════════════════════════════════
user_seen_movies = df_train.groupby("userId")["movieId"].apply(set).to_dict()

@st.cache_data(ttl=300, show_spinner=False)
def recommend_movies_cached(uid, model_name, k, min_imdb, alpha, beta, exclude_seen):
    """
    Full recommendation pipeline, cached per unique (uid, model, params) combo.
    Returns a DataFrame of top-k recommendations.

    OPT 3: Smart candidate selection — popular movies first + random diversity.
    OPT 1+2: Batch SVD + vectorized ItemCF instead of per-item loops.
    """
    # ── Candidate selection ──────────────────────────────────────────────────
    seen = user_seen_movies.get(uid, set()) if exclude_seen else set()
    candidates = [m for m in movie_ids_all if m not in seen]

    if len(candidates) > 2000:
        # 1500 most popular + 500 random for diversity
        popular_cands = [m for m in popular_movies_ordered if m in set(candidates)][:1500]
        remaining     = list(set(candidates) - set(popular_cands))
        rng           = np.random.RandomState(uid % 100000)
        random_cands  = rng.choice(remaining, min(500, len(remaining)), replace=False).tolist()
        candidates    = popular_cands + random_cands

    cand_tuple = tuple(candidates)  # hashable for inner caches

    # ── Batch scoring by model ───────────────────────────────────────────────
    # Force Content-Based for cold-start users
    effective_model = model_name
    if is_cold_start(uid) and model_name not in ("📄 Content-Based",):
        effective_model = "📄 Content-Based"

    if effective_model == "🔬 SVD":
        score_map = predict_svd_batch(uid, cand_tuple)

    elif effective_model == "🔗 ItemCF":
        score_map = predict_itemcf_batch(uid, cand_tuple)

    elif effective_model == "📄 Content-Based":
        score_map = predict_content_batch(uid, cand_tuple)

    elif effective_model == "⚡ Hybrid SVD + ItemCF":
        svd_scores  = predict_svd_batch(uid, cand_tuple)
        icf_scores  = predict_itemcf_batch(uid, cand_tuple)
        score_map   = {}
        for m in candidates:
            sv = svd_scores.get(m, np.nan)
            ic = icf_scores.get(m, np.nan)
            if np.isnan(ic):
                score_map[m] = sv
            elif np.isnan(sv):
                score_map[m] = ic
            else:
                score_map[m] = float(alpha * sv + (1 - alpha) * ic)

    elif effective_model == "🏆 Hybrid + IMDb Boost":
        svd_scores = predict_svd_batch(uid, cand_tuple)
        icf_scores = predict_itemcf_batch(uid, cand_tuple)
        score_map  = {}
        for m in candidates:
            sv = svd_scores.get(m, np.nan)
            ic = icf_scores.get(m, np.nan)
            base = sv if np.isnan(ic) else (ic if np.isnan(sv) else float(alpha * sv + (1 - alpha) * ic))
            if not np.isnan(base):
                imdb_n = imdb_norm_map.get(m, 0.5)
                score_map[m] = float(np.clip(base + beta * (imdb_n - 0.5), R_MIN, R_MAX))

    elif effective_model == "🧠 Adaptive Hybrid + IMDb Boost":
        alpha_u    = adaptive_alpha_fn(uid)
        svd_scores = predict_svd_batch(uid, cand_tuple)
        icf_scores = predict_itemcf_batch(uid, cand_tuple)
        score_map  = {}
        for m in candidates:
            sv   = svd_scores.get(m, np.nan)
            ic   = icf_scores.get(m, np.nan)
            base = sv if np.isnan(ic) else (ic if np.isnan(sv) else float(alpha_u * sv + (1 - alpha_u) * ic))
            if not np.isnan(base):
                imdb_n = imdb_norm_map.get(m, 0.5)
                score_map[m] = float(np.clip(base + beta * (imdb_n - 0.5), R_MIN, R_MAX))

    else:
        score_map = predict_svd_batch(uid, cand_tuple)

    # ── Build result DataFrame ───────────────────────────────────────────────
    scored = [(m, s) for m, s in score_map.items() if not np.isnan(s)]
    scored.sort(key=lambda x: x[1], reverse=True)

    top_ids   = [m for m, _ in scored[: k * 3]]
    score_ref = {m: s for m, s in scored}

    df_rec = df_movies[df_movies["movieId"].isin(top_ids)].copy()
    df_rec["pred_score"] = df_rec["movieId"].map(score_ref)
    df_rec = df_rec.dropna(subset=["pred_score"])

    if min_imdb > 0 and "imdb_score" in df_rec.columns:
        df_rec = df_rec[df_rec["imdb_score"] >= min_imdb]

    return df_rec.sort_values("pred_score", ascending=False).head(k).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
        <span style='font-size:3rem'>🎬</span>
        <h2 style='color:#e94560; margin:8px 0 4px; font-size:1.4rem'>Movie Recommender</h2>
        <p style='color:#8888aa; font-size:0.8rem'>Hybrid SVD + ItemCF + IMDb Boost</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    all_user_ids = sorted(user_to_idx.keys())
    user_id = st.selectbox(
        "👤 Chọn User ID",
        options=all_user_ids,
        index=0,
        help="Chọn user để nhận gợi ý phim",
    )

    model_choice = st.radio(
        "🤖 Chọn Model",
        options=[
            "🏆 Hybrid + IMDb Boost",
            "🧠 Adaptive Hybrid + IMDb Boost",
            "⚡ Hybrid SVD + ItemCF",
            "🔬 SVD",
            "🔗 ItemCF",
            "📄 Content-Based",
            "📊 Compare All",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("**⚙️ Cài Đặt Nâng Cao**")

    k_recs   = st.slider("🎯 Số phim gợi ý", 5, 20, 10)
    min_imdb = st.slider("⭐ IMDb Score tối thiểu", 0.0, 9.0, 0.0, 0.5,
                         help="Chỉ hiển thị phim có IMDb score ≥ giá trị này")

    if model_choice in ("⚡ Hybrid SVD + ItemCF", "🏆 Hybrid + IMDb Boost"):
        alpha_val = st.slider("α (SVD weight)", 0.0, 1.0, float(BEST_ALPHA), 0.1,
                              help=f"Tỉ lệ SVD trong Hybrid. Tối ưu: {BEST_ALPHA}")
    else:
        alpha_val = BEST_ALPHA

    if model_choice == "🏆 Hybrid + IMDb Boost":
        beta_val = st.slider("β (IMDb boost)", 0.0, 1.0, float(BEST_BETA), 0.1,
                             help=f"Sức mạnh của IMDb Boost. Tối ưu: {BEST_BETA}")
    else:
        beta_val = BEST_BETA

    exclude_seen_flag = st.checkbox("Ẩn phim đã xem", value=True)

    st.markdown("---")
    cold_start_flag = is_cold_start(user_id)
    if cold_start_flag:
        st.warning(f"❄️ Cold-Start User\n\n< {COLD_THR} ratings → tự động dùng Content-Based")
    else:
        hist_n = get_user_rating_count(user_id)
        st.success(f"✅ Active User\n\n{hist_n} ratings trong train set")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='padding: 30px 0 10px;'>
    <h1 style='color:#ffffff; font-size:2.5rem; font-weight:800; margin:0;'>
        🎬 Movie Recommendation System
    </h1>
    <p style='color:#8888aa; font-size:1.05rem; margin-top:6px;'>
        Hybrid SVD + ItemCF · IMDb Score Boost · Content-Based Cold-Start · ml-25m Dataset
    </p>
</div>
""", unsafe_allow_html=True)


# ─── Metrics Dashboard ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Model Performance Dashboard</div>', unsafe_allow_html=True)

models_display = {
    "SVD": "svd", "ItemCF": "itemcf", "Content": "content",
    "Hybrid": "hybrid", "Hybrid+Boost": "hybrid_boosted",
    "Adaptive Hybrid+Boost": "hybrid_adaptive_boosted",
}

cols_metrics = st.columns(len(models_display))
for col, (display_name, key) in zip(cols_metrics, models_display.items()):
    with col:
        rmse_v       = RMSE.get(key, 0)
        p10_v        = P10.get(key, 0)
        r10_v        = R10.get(key, 0)
        is_best_rmse = key == min(RMSE, key=RMSE.get) if RMSE else False
        border       = "border: 2px solid #e94560 !important;" if is_best_rmse else ""
        st.markdown(f"""
        <div class="metric-card" style="{border}">
            <div class="metric-title">{display_name}</div>
            <div class="metric-value {'metric-good' if is_best_rmse else 'metric-mid'}">
                {rmse_v:.3f}
            </div>
            <div style='color:#8888aa; font-size:0.72rem; margin-top:4px;'>RMSE</div>
            <hr style='margin:8px 0; border-color:#ffffff15'>
            <div style='display:flex; justify-content:space-between; font-size:0.78rem;'>
                <span style='color:#aaaacc'>P@10: <b style='color:#2ecc71'>{p10_v:.4f}</b></span>
                <span style='color:#aaaacc'>R@10: <b style='color:#3498db'>{r10_v:.4f}</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── Best Params Banner ────────────────────────────────────────────────────────
best_svd_p = meta.get("best_svd_params", {})
st.markdown(f"""
<div style='background: linear-gradient(135deg, #533483 0%, #0f3460 100%);
     border-radius: 12px; padding: 14px 20px; margin-bottom: 20px;
     display: flex; gap: 30px; flex-wrap: wrap;'>
    <div><span style='color:#c39bd3; font-size:0.78rem'>BEST SVD PARAMS</span><br>
         <b style='color:#fff'>n_factors={best_svd_p.get("n_factors","?")} | n_epochs={best_svd_p.get("n_epochs","?")} |
         lr={best_svd_p.get("lr_all","?")} | reg={best_svd_p.get("reg_all","?")}</b></div>
    <div><span style='color:#c39bd3; font-size:0.78rem'>HYBRID PARAMS</span><br>
         <b style='color:#fff'>α (SVD weight) = {BEST_ALPHA} | β (IMDb boost) = {BEST_BETA}</b></div>
    <div><span style='color:#c39bd3; font-size:0.78rem'>COLD-START</span><br>
         <b style='color:#fff'>Content-Based khi &lt; {COLD_THR} ratings</b></div>
</div>
""", unsafe_allow_html=True)


# ─── Recommendation Renderer ───────────────────────────────────────────────────
def render_recommendations(uid, model_name, k, min_imdb, alpha, beta, exclude_seen):
    """Fetch (from cache or compute) and render movie recommendation cards."""

    # Resolve effective model label for display
    if is_cold_start(uid) and model_name not in ("📄 Content-Based", "📊 Compare All"):
        st.info(f"❄️ User {uid} là Cold-Start user → tự động dùng **Content-Based**")
        effective_model  = "📄 Content-Based"
        display_label    = "Content-Based (Cold-Start)"
    else:
        effective_model = model_name
        display_label   = model_name

    with st.spinner(f"Computing recommendations ({display_label})..."):
        df_rec = recommend_movies_cached(
            uid, effective_model, k, min_imdb, alpha, beta, exclude_seen
        )

    if df_rec.empty:
        st.warning("Không có phim nào thỏa mãn bộ lọc. Hãy giảm IMDb threshold.")
        return

    cols_per_row = 3
    for row_start in range(0, len(df_rec), cols_per_row):
        cols = st.columns(cols_per_row)
        for col_idx, col in enumerate(cols):
            idx = row_start + col_idx
            if idx >= len(df_rec):
                break
            row     = df_rec.iloc[idx]
            title   = row.get("title", "N/A")
            genres  = row.get("genres_str", row.get("genres", ""))
            pred_sc = row.get("pred_score", 0)
            imdb_sc = row.get("imdb_score", None)
            imdb_badge = f'<span class="badge-imdb">⭐ IMDb {imdb_sc:.2f}</span>' if pd.notna(imdb_sc) else ""
            with col:
                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">#{idx+1} {title}</div>
                    <div class="movie-genres">{genres}</div>
                    <div class="movie-score-row">
                        <span class="badge-pred">🎯 {pred_sc:.2f} / {R_MAX:.1f}</span>
                        {imdb_badge}
                    </div>
                    <div style='margin-top:8px;'>
                        <span class="badge-model">{display_label.replace("🏆 ","").replace("⚡ ","").replace("🧠 ","")}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─── Recommendations Section ───────────────────────────────────────────────────
st.markdown(f'<div class="section-header">🎯 Gợi Ý Cho User {user_id}</div>', unsafe_allow_html=True)

if model_choice == "📊 Compare All":
    st.markdown("#### So sánh tất cả 6 mô hình:")
    tab_names  = ["🏆 Hybrid+Boost", "🧠 Adaptive Hybrid+Boost", "⚡ Hybrid", "🔬 SVD", "🔗 ItemCF", "📄 Content"]
    tab_models = [
        "🏆 Hybrid + IMDb Boost",
        "🧠 Adaptive Hybrid + IMDb Boost",
        "⚡ Hybrid SVD + ItemCF",
        "🔬 SVD",
        "🔗 ItemCF",
        "📄 Content-Based",
    ]
    tabs = st.tabs(tab_names)
    for tab, model_name in zip(tabs, tab_models):
        with tab:
            render_recommendations(user_id, model_name, k_recs, min_imdb,
                                   alpha_val, beta_val, exclude_seen_flag)
else:
    render_recommendations(user_id, model_choice, k_recs, min_imdb,
                           alpha_val, beta_val, exclude_seen_flag)


# ─── User History ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📚 Lịch Sử Xem</div>', unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_user_history(uid):
    hist = df_train[df_train["userId"] == uid].copy()
    hist = hist.merge(df_movies[["movieId", "title", "genres_str"]], on="movieId", how="left")
    return hist.sort_values("rating", ascending=False)

user_hist = get_user_history(user_id)

if user_hist.empty:
    st.info("Không có lịch sử xem trong train set.")
else:
    with st.expander(f"▶ {len(user_hist)} phim đã xem (Click để xem)"):
        cols_h = st.columns(2)
        for i, (_, row) in enumerate(user_hist.head(20).iterrows()):
            col   = cols_h[i % 2]
            stars = "⭐" * int(row["rating"])
            with col:
                st.markdown(f"""
                <div class="history-item">
                    <b>{row.get("title", "N/A")}</b><br>
                    <span style='color:#8888aa; font-size:0.78rem'>{row.get("genres_str","")}</span><br>
                    <span style='color:#e2b04a'>{stars}</span>
                    <span style='color:#555566; font-size:0.78rem'> ({row["rating"]})</span>
                </div>
                """, unsafe_allow_html=True)


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style='text-align:center; color:#555577; font-size:0.78rem; padding:10px;'>
    🎬 Movie Recommendation System · MovieLens ml-25m ·
    Hybrid SVD+ItemCF+IMDb Boost · Data Mining Project
</div>
""", unsafe_allow_html=True)