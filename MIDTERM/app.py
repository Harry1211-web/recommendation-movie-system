import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Suppress pandas dtype warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore')

# Disable StringDtype completely
pd.options.mode.string_storage = "python"

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD MODELS VÀ DỮ LIỆU
# ============================================================
@st.cache_resource
def load_models():
    """Load tất cả các model đã train"""
    try:
        metadata = joblib.load('models/model_metadata.pkl')
    except Exception as e:
        print(f"Lỗi load metadata: {e}")
        return None, None, None, None, None
    
    # Load df_train từ CSV
    try:
        df_train = pd.read_csv('models/df_train.csv')
    except Exception as e:
        print(f"Lỗi load df_train: {e}")
        df_train = None
    
    # Load các models
    try:
        usercf_model = joblib.load('models/usercf_model.pkl')
    except Exception as e:
        print(f"Lỗi load UserCF model: {e}")
        usercf_model = None
    
    try:
        itemcf_model = joblib.load('models/itemcf_model.pkl')
    except Exception as e:
        print(f"Lỗi load ItemCF model: {e}")
        itemcf_model = None
    
    try:
        svd_model = joblib.load('models/svd_model.pkl')
    except Exception as e:
        print(f"Lỗi load SVD model: {e}")
        svd_model = None

    try:
        content_model = joblib.load('models/content_model.pkl')
    except Exception as e:
        print(f"Lỗi load Content-Based model: {e}")
        content_model = None
    
    try:
        df_movies = pd.read_csv('models/movies_metadata_for_testing.csv')
    except Exception as e:
        print(f"Lỗi load movies metadata: {e}")
        df_movies = None
    
    # Bổ sung metadata build tự động nếu thiếu thông tin map
    if metadata is None:
        metadata = {}

    if 'user_to_idx' not in metadata and usercf_model is not None and 'user_to_idx' in usercf_model:
        metadata['user_to_idx'] = usercf_model['user_to_idx']
    if 'movie_to_idx' not in metadata and usercf_model is not None and 'movie_to_idx' in usercf_model:
        metadata['movie_to_idx'] = usercf_model['movie_to_idx']

    # Build idx_to_* maps nếu cần
    if 'idx_to_user' not in metadata and 'user_to_idx' in metadata:
        metadata['idx_to_user'] = {v: k for k, v in metadata['user_to_idx'].items()}
    if 'idx_to_movie' not in metadata and 'movie_to_idx' in metadata:
        metadata['idx_to_movie'] = {v: k for k, v in metadata['movie_to_idx'].items()}

    if 'user_ids' not in metadata and 'user_to_idx' in metadata:
        metadata['user_ids'] = list(metadata['user_to_idx'].keys())
    if 'movie_ids' not in metadata and 'movie_to_idx' in metadata:
        metadata['movie_ids'] = list(metadata['movie_to_idx'].keys())

    return metadata, usercf_model, itemcf_model, svd_model, content_model, df_movies, df_train

# Load models
metadata, usercf_model, itemcf_model, svd_model, content_model, df_movies, df_train = load_models()

# ============================================================
# BUILD MOVIE LOOKUP CACHE (optimize performance)
# ============================================================
@st.cache_data
def build_movie_lookup():
    """Build a dictionary for fast movie lookups by movieId"""
    if df_movies is None or df_movies.empty:
        return {}
    
    movie_dict = {}
    for _, row in df_movies.iterrows():
        movie_id = row['movieId']
        if movie_id not in movie_dict:  # Lấy lần đầu (unique)
            movie_dict[movie_id] = {
                'title': row.get('title', 'N/A'),
                'genres': row.get('genres', 'N/A'),
                'release_year': str(row.get('release_year', 'N/A')),
                'tmdbId': str(row.get('tmdbId', 'N/A')),
            }
    return movie_dict

movie_dict = build_movie_lookup()

# Check if all models loaded successfully
if metadata is None or usercf_model is None or itemcf_model is None or svd_model is None or content_model is None or df_movies is None or df_train is None:
    st.error("Lỗi: Không thể load các model. Vui lòng kiểm tra file trong thư mục 'models/'")
    st.stop()

# Extract mappings
user_to_idx = metadata['user_to_idx']
idx_to_user = metadata['idx_to_user']
movie_to_idx = metadata['movie_to_idx']
idx_to_movie = metadata['idx_to_movie']
user_ids = metadata['user_ids']
movie_ids = metadata['movie_ids']

# ============================================================
# ĐỊNH NGHĨA CÁC HÀM PREDICT
# ============================================================
def predict_usercf(user_id, movie_id, k=20):
    """Predict rating using UserCF"""
    user_sim = usercf_model['user_sim']
    R = usercf_model['R']
    # user_to_idx e movie_to_idx são global variables carregadas de metadata
    
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return np.nan
    
    uidx = user_to_idx[user_id]
    midx = movie_to_idx[movie_id]
    sims = user_sim[uidx]
    ratings = R[:, midx].toarray().reshape(-1)
    
    mask = ratings > 0
    if mask.sum() == 0:
        return np.nan
    
    sims = sims[mask]
    ratings = ratings[mask]
    top_k_idx = np.argsort(sims)[-k:]
    sim_k = sims[top_k_idx]
    rating_k = ratings[top_k_idx]
    
    if sim_k.sum() == 0:
        return np.nan
    
    return np.dot(sim_k, rating_k) / sim_k.sum()

def predict_itemcf(user_id, movie_id, k=20):
    """Predict rating using ItemCF"""
    item_sim = itemcf_model['item_sim']
    R = itemcf_model['R']
    # user_to_idx e movie_to_idx são global variables carregadas de metadata
    
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return np.nan
    
    uidx = user_to_idx[user_id]
    midx = movie_to_idx[movie_id]
    user_ratings = R[uidx].toarray().reshape(-1)
    mask = user_ratings > 0
    
    if mask.sum() == 0:
        return np.nan
    
    sims = item_sim[midx, mask]
    rating_k = user_ratings[mask]
    top_k_idx = np.argsort(sims)[-k:]
    sim_k = sims[top_k_idx]
    rating_k_selected = rating_k[top_k_idx]
    
    if sim_k.sum() == 0:
        return np.nan
    
    return np.dot(sim_k, rating_k_selected) / sim_k.sum()

def predict_svd(user_id, movie_id):
    """Predict rating using Surprise SVD"""
    # svd_model is now a Surprise SVD object directly loaded via joblib/pickle
    if user_id not in user_ids or movie_id not in movie_ids:
        return np.nan
    
    try:
        return svd_model.predict(user_id, movie_id).est
    except Exception:
        return np.nan

# ============================================================
# Content-Based predict
# ============================================================
def predict_contentbased(user_id, movie_id):
    """
    Predict rating using Content-Based Recommendation
    - Xử lý cả sparse matrix (CSR) và dense numpy arrays
    - ✅ Optimized for memory efficiency
    """
    if content_model is None:
        return np.nan
    user_profile = content_model.get('user_profile', {})
    movie_vecs = content_model.get('movie_vec', {})

    if user_id not in user_profile or movie_id not in movie_vecs:
        return np.nan

    profile = user_profile[user_id]
    if profile is None:
        return np.nan

    # Xử lý cả sparse và dense formats
    movie_vec = movie_vecs[movie_id]
    
    if hasattr(profile, 'todense'):  # Profile là sparse
        sim = cosine_similarity(profile.reshape(1, -1), movie_vec)[0, 0]
    elif hasattr(movie_vec, 'todense'):  # Movie vec là sparse
        sim = cosine_similarity(profile.reshape(1, -1), movie_vec)[0, 0]
    else:  # Cả hai đều dense
        sim = cosine_similarity(profile.reshape(1, -1), movie_vec.reshape(1, -1))[0, 0]
    
    return np.clip(sim * 5.0, 0.0, 5.0)

# ============================================================
# HYBRID MODEL predict (SVD + Content-Based)
# ============================================================
def predict_hybrid(user_id, movie_id, alpha=0.7):
    """
    Kết hợp SVD (Collaborative) và Content-Based
    SVD đóng vai trò chủ đạo (alpha), Content-Based bù đắp cold-start/long-tail (1-alpha)
    """
    svd_score = predict_svd(user_id, movie_id)
    cb_score = predict_contentbased(user_id, movie_id)
    
    if np.isnan(svd_score) and np.isnan(cb_score):
        return np.nan
    
    if np.isnan(cb_score): 
        return svd_score
    if np.isnan(svd_score):
        return cb_score
        
    return alpha * svd_score + (1.0 - alpha) * cb_score

# ============================================================
# HÀM TÍNH CONFIDENCE SCORE
# ============================================================
def get_itemcf_details(user_id, movie_id, k=20):
    """
    Lấy chi tiết dự đoán từ ItemCF bao gồm:
    - Predicted rating
    - Confidence score (dựa trên độ mạnh của neighbors)
    - Number of neighbors
    - Rating range từ neighbors
    """
    item_sim = itemcf_model['item_sim']
    R = itemcf_model['R']
    # user_to_idx e movie_to_idx são global variables carregadas de metadata
    
    if user_id not in user_to_idx or movie_id not in movie_to_idx:
        return None, None, None, None, None
    
    uidx = user_to_idx[user_id]
    midx = movie_to_idx[movie_id]
    user_ratings = R[uidx].toarray().reshape(-1)
    mask = user_ratings > 0
    
    if mask.sum() == 0:
        return None, None, None, None, None
    
    sims = item_sim[midx, mask]
    rating_k = user_ratings[mask]
    top_k_idx = np.argsort(sims)[-k:]
    sim_k = sims[top_k_idx]
    rating_k_selected = rating_k[top_k_idx]
    
    if sim_k.sum() == 0:
        return None, None, None, None, None
    
    pred = np.dot(sim_k, rating_k_selected) / sim_k.sum()
    
    # Tính confidence score dựa trên:
    # 1. Trung bình độ tương đồng của neighbors
    # 2. Đồng thuận giữa neighbors (variance)
    avg_sim = sim_k.mean()
    rating_variance = rating_k_selected.var()
    confidence = avg_sim * (1 - min(rating_variance / 6.25, 1))  # Max variance = 6.25 (5-0=5, var=25/4)
    
    return pred, confidence, len(sim_k), rating_k_selected.min(), rating_k_selected.max()

# ============================================================
# CONTENT-BASED MODEL (for cold-start users)
# ============================================================
@st.cache_resource
@st.cache_data
def get_available_content_movies():
    """
    Lấy danh sách movieIds có sẵn features cho Content-Based model
    """
    from collections import defaultdict
    
    movie_tags = defaultdict(lambda: [])
    for _, row in df_movies.iterrows():
        movie_id = row['movieId']
        tag = str(row.get('tag', '')).strip()
        if tag and tag != 'nan':
            movie_tags[movie_id].append(tag)
    
    return set(movie_tags.keys())

def build_content_features():
    """
    Xây dựng feature vectors cho từng phim dựa trên tags và genres.
    Returns: movie_features (dict), vectorizer (TfidfVectorizer)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from collections import defaultdict
    
    # Nhóm tags theo movieId
    movie_tags = defaultdict(lambda: [])
    for _, row in df_movies.iterrows():
        movie_id = row['movieId']
        tag = str(row.get('tag', '')).strip()
        if tag and tag != 'nan':
            movie_tags[movie_id].append(tag)
    
    # Tạo TF-IDF vectorizer
    all_tags = [' '.join(movie_tags[m_id]) if movie_tags[m_id] else 'unknown' for m_id in sorted(movie_tags.keys())]
    vectorizer = TfidfVectorizer(max_features=100, lowercase=True, stop_words='english')
    tag_vectors = vectorizer.fit_transform(all_tags)
    
    # Tạo dict ánh xạ movieId -> feature vector
    movie_ids_list = sorted(movie_tags.keys())
    movie_features = {}
    for i, movie_id in enumerate(movie_ids_list):
        movie_features[movie_id] = tag_vectors[i]
    
    return movie_features, vectorizer, movie_ids_list

def recommend_contentbased(preferred_movie_ids, k=10, exclude_movie_ids=None):
    """
    Gợi ý phim dựa trên nội dung (tags + genres).
    Sử dụng cho cold-start users hoặc content-based exploration.
    
    Args:
        preferred_movie_ids: List of movieIds that user likes
        k: Số lượng gợi ý
        exclude_movie_ids: Movies to exclude (user đã rated)
    
    Returns:
        DataFrame với cột [movieId, title, genres, sim_score]
    """
    movie_features, vectorizer, movie_ids_list = build_content_features()
    
    if not preferred_movie_ids:
        return pd.DataFrame()
    
    # Filter movies that have features
    valid_preferred_ids = [m_id for m_id in preferred_movie_ids if m_id in movie_features]
    
    if not valid_preferred_ids:
        return pd.DataFrame()
    
    # Tính trung bình feature vector của những phim user thích
    first_valid_id = valid_preferred_ids[0]
    user_preference = np.zeros(movie_features[first_valid_id].shape[1])
    count = 0
    for m_id in valid_preferred_ids:
        if m_id in movie_features:
            user_preference += movie_features[m_id].toarray().flatten()
            count += 1
    
    if count == 0:
        return pd.DataFrame()
    
    user_preference /= count
    user_preference = user_preference.reshape(1, -1)
    
    # Tính similarity với tất cả phim
    scores = []
    for m_id in movie_ids_list:
        if exclude_movie_ids and m_id in exclude_movie_ids:
            continue
        if m_id in preferred_movie_ids:  # Bỏ qua những phim đã like
            continue
        
        sim = cosine_similarity(user_preference, movie_features[m_id].toarray())[0][0]
        scores.append((m_id, sim))
    
    # Sort và lấy top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    top_scores = scores[:k]
    
    # Tạo DataFrame kết quả
    result_data = []
    for rank, (m_id, sim_score) in enumerate(top_scores, 1):
        movie_info = df_movies[df_movies['movieId'] == m_id].drop_duplicates('movieId').iloc[0]
        result_data.append({
            'Rank': rank,
            'movieId': m_id,
            'Title': movie_info['title'],
            'Genres': movie_info.get('genres', 'N/A'),
            'Score': round(sim_score, 3)
        })
    
    return pd.DataFrame(result_data) if result_data else pd.DataFrame()

# ============================================================
# HÀM GỢI ÝN PHIM
# ============================================================
def recommend_movies(user_id, k=10, model_type='itemcf', threshold=2.5, confidence_threshold=0.0):
    """Gợi ý top-K phim cho user"""
    
    if user_id not in user_to_idx:
        return None, f"User {user_id} không tìm thấy trong dữ liệu huấn luyện"
    
    # Lấy danh sách phim đã rated (df_train được load từ models/df_train.csv)
    user_rated_movies = set(df_train[df_train['userId'] == user_id]['movieId'].values)
    
    # Chọn hàm predict
    if model_type.lower() == 'usercf':
        predict_fn = predict_usercf
    elif model_type.lower() == 'itemcf':
        predict_fn = predict_itemcf
    elif model_type.lower() == 'svd':
        predict_fn = predict_svd
    elif model_type.lower() in ['content', 'content-based', 'content-based (cold-start)', 'content-based (cold-start)'.lower()]:
        predict_fn = predict_contentbased
    elif model_type.lower() in ['hybrid', 'hybrid (tối ưu nhất)'.lower()]:
        predict_fn = predict_hybrid
    else:
        return None, f"Loại model không hợp lệ: {model_type}"
    
    # Tính score cho tất cả phim chưa rated
    scores = []
    for movie_id in movie_ids:
        if movie_id in user_rated_movies:
            continue  # Bỏ qua phim đã rated
        
        # Đối với ItemCF, lấy confidence details
        if model_type.lower() == 'itemcf':
            pred, confidence, num_neighbors, min_rating, max_rating = get_itemcf_details(
                user_id, movie_id, k=20
            )
            if pred is not None and not np.isnan(pred) and pred >= threshold and confidence >= confidence_threshold:
                scores.append((movie_id, pred, confidence, num_neighbors, min_rating, max_rating))
        else:
            score = predict_fn(user_id, movie_id)
            if not np.isnan(score) and score >= threshold:
                # Thêm placeholder values cho non-ItemCF models
                scores.append((movie_id, score, None, None, None, None))
    
    # Sort và lấy top-k
    scores.sort(key=lambda x: x[1], reverse=True)
    top_movies = scores[:k]
    
    # Tạo DataFrame kết quả
    results = []
    for idx, item in enumerate(top_movies, 1):
        if model_type.lower() == 'itemcf':
            movie_id, score, confidence, num_neighbors, min_rating, max_rating = item
        else:
            movie_id, score, _, _, _, _ = item
            confidence, num_neighbors, min_rating, max_rating = None, None, None, None
        
        # Use cached movie_dict for faster lookup
        if movie_id in movie_dict:
            movie_info = movie_dict[movie_id]
        else:
            movie_info = {'title': 'N/A', 'genres': 'N/A', 'release_year': 'N/A'}
        
        # Tạo confidence badge
        if confidence is not None:
            if confidence >= 0.7:
                conf_badge = "High"
            elif confidence >= 0.4:
                conf_badge = "Medium"
            else:
                conf_badge = "Low"
        else:
            conf_badge = "N/A"
        
        result_dict = {
            'Rank': idx,
            'Movie ID': movie_id,
            'Title': movie_info.get('title', 'N/A'),
            'Genres': movie_info.get('genres', 'N/A'),
            'Release Year': movie_info.get('release_year', 'N/A'),
            'Predicted Rating': f"{score:.2f}/5.0",
            'Score': score
        }
        
        # Thêm ItemCF details nếu có
        if confidence is not None:
            result_dict['Confidence'] = conf_badge
            result_dict['Conf Score'] = f"{confidence:.3f}"
            result_dict['Neighbors'] = int(num_neighbors) if num_neighbors else 0
            result_dict['Rating Range'] = f"{min_rating:.1f}-{max_rating:.1f}" if min_rating and max_rating else "N/A"
        
        results.append(result_dict)
    
    return pd.DataFrame(results) if results else None, None

# ============================================================
# UI CHÍNH
# ============================================================
st.title("Movie Recommendation System")

# Hướng dẫn sử dụng (Đầu trang)
with st.expander("Hướng dẫn sử dụng - Click để xem"):
    st.write("""
    ### Cách sử dụng ứng dụng:
    
    1. **Chọn User ID** - Chọn một user từ dropdown (có 610 users có sẵn)
    2. **Chọn Model** - Có 4 model để chọn hoặc so sánh cả 4: ItemCF, SVD, UserCF, Content-Based
    3. **Số phim gợi ý** - Điều chỉnh từ 5 đến 20 phim
    4. **Threshold rating** - Chỉ gợi ý phim có predicted rating ≥ ngưỡng
    5. **Confidence threshold** (ItemCF only) - Filter gợi ý theo độ tin cậy
    
    ### Các Model:
    
    **ItemCF (Item-based Collaborative Filtering)**
    - RMSE: 0.9499 (THẤP NHẤT) | MAE: 0.7146 (THẤP NHẤT)
    - Dùng: Dự đoán rating chính xác
    - Ưu điểm: Ổn định, dễ triển khai, nhanh
    - Cách hoạt động: Tìm phim tương tự với những phim user đã thích
    
    **SVD (Matrix Factorization)**
    - Precision@10: 0.0030 (TỐT NHẤT) | Recall@10: 0.0295 (TỐT NHẤT)
    - Dùng: Gợi ý phim mới, tìm phim liên quan (top-K recommendations)
    - Ưu điểm: Học embedding ẩn, xử lý sparsity tốt
    - Cách hoạt động: Phân rã user-item matrix thành latent factors
    
    **UserCF (User-based Collaborative Filtering)**
    - Hiệu năng trung bình
    - Dùng: Tìm người dùng có hành vi tương tự
    - Ưu điểm: Hiểu user behavior patterns tốt
    - Cách hoạt động: Tìm user tương tự và gợi ý phim họ thích
    
    **Content-Based (Dựa vào đặc trưng phim)**
    - Tốt cho cold-start problem (người dùng/phim mới)
    - Dùng: Xử lý user/phim mới, không cần rating history
    - Ưu điểm: Không bị ảnh hưởng bởi data sparsity
    - Cách hoạt động: Tính user profile từ trung bình feature của phim đã rate, dùng cosine similarity
    
    **Hybrid (SVD + Content-Based) [Đề Xuất]**
    - Là thuật toán hoàn thiện nhất trong ứng dụng
    - Kết hợp dự đoán của cả Matrix Factorization mạnh mẽ và Content-Based
    - Cứu cánh những bộ phim thuộc vùng đuôi dài (Long-tail)
    
    ### Metrics:
    - **RMSE**: Root Mean Square Error (độ lệch căn bậc hai) - càng thấp càng tốt
    - **MAE**: Mean Absolute Error (sai số tuyệt đối trung bình) - càng thấp càng tốt
    - **Precision@10**: % phim gợi ý mà user thực sự thích - càng cao càng tốt
    - **Recall@10**: % phim mà user thích được tìm thấy trong top-10 gợi ý - càng cao càng tốt
    
    ### Khuyến nghị:
    - **Accuracy**: Dùng **ItemCF** (RMSE/MAE thấp nhất)
    - **Ranking**: Dùng **SVD** (Precision/Recall tốt nhất)
    - **Cold-Start**: Dùng **Content-Based** (cho user/phim mới)
    - **Hybrid**: Kết hợp 2+ models để robustness tốt hơn
    """)

st.markdown("---")

# Sidebar: Chọn model và users
with st.sidebar:
    st.header("Cấu hình")
    
    # Chọn user
    selected_user = st.selectbox(
        "Chọn User ID:",
        options=sorted(list(user_ids)),
        help="Chọn user để nhận gợi ý phim"
    )
    
    # Chọn model
    model_choice = st.selectbox(
        "Chọn Model:",
        options=['Hybrid (Tối Ưu Nhất)', 'ItemCF', 'SVD', 'UserCF', 'Content-Based (Cold-Start)', 'Compare All'],
        help="Hybrid kết hợp mạnh mẽ SVD & Content-Based. ItemCF: Prediction error thấp | SVD: Ranking chất lượng tốt | UserCF: Tìm user tương tự | Content-Based: Cho user mới"
    )
    
    # Nếu chọn Content-Based, cho phép chọn preferred movies
    preferred_movies = []
    if model_choice == 'Content-Based (Cold-Start)':
        st.info("Content-Based Mode: Chọn những phim bạn thích để khám phá phim tương tự!")
        
        # Lấy danh sách phim có sẵn features
        available_movies = get_available_content_movies()
        
        # Tìm kiếm và chọn phim
        search_title = st.text_input("Tìm kiếm phim (nhập kí tự để tìm):", placeholder="Ví dụ: Avatar, Titanic, ...")
        
        # Hiển thị kết quả tìm kiếm real-time
        if search_title and len(search_title) > 0:
            matching_movies = df_movies[df_movies['title'].str.contains(search_title, case=False, na=False)]
            # Filter để chỉ lấy phim có features
            matching_movies = matching_movies[matching_movies['movieId'].isin(available_movies)]
            unique_movies = matching_movies.drop_duplicates('movieId').sort_values('title')
            
            if not unique_movies.empty:
                st.write(f"**Phim tìm được ({len(unique_movies)}):**")
                
                # Hiển thị các phim tìm được dưới dạng buttons
                for idx, (_, movie) in enumerate(unique_movies.head(20).iterrows()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{movie['title']}")
                    with col2:
                        btn_key = f"add_search_{movie['movieId']}"
                        if st.button("➕", key=btn_key):
                            st.session_state.setdefault('preferred_movies_list', [])
                            if movie['movieId'] not in st.session_state['preferred_movies_list']:
                                st.session_state['preferred_movies_list'].append(movie['movieId'])
                                st.success(f"Đã thêm {movie['title']}!")
                                st.rerun()
                            else:
                                st.warning("Phim này đã được thêm rồi!")
            else:
                st.warning(f"Không tìm thấy phim có chứa '{search_title}' (hoặc phim chưa có tags)")
        else:
            # Nếu chưa tìm kiếm, hiển thị gợi ý duyệt phim (top 20)
            with st.expander("Gợi ý 20 phim phổ biến", expanded=False):
                # Lấy phim duy nhất có sẵn features
                all_unique_movies = df_movies[df_movies['movieId'].isin(available_movies)]
                all_unique_movies = all_unique_movies.drop_duplicates('movieId').sort_values('title').head(20)
                
                # Hiển thị từng phim với nút thêm
                for idx, (_, movie) in enumerate(all_unique_movies.iterrows()):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"{movie['title']}")
                    with col2:
                        btn_key = f"add_browse_{movie['movieId']}"
                        if st.button("➕", key=btn_key):
                            st.session_state.setdefault('preferred_movies_list', [])
                            if movie['movieId'] not in st.session_state['preferred_movies_list']:
                                st.session_state['preferred_movies_list'].append(movie['movieId'])
                                st.success(f"Đã thêm {movie['title']}!")
                                st.rerun()
                            else:
                                st.warning("Phim này đã được thêm rồi!")
        
        # Hiển thị danh sách phim đã chọn
        if 'preferred_movies_list' in st.session_state and st.session_state['preferred_movies_list']:
            st.write("**Phim đã chọn:**")
            for m_id in st.session_state['preferred_movies_list']:
                col1, col2 = st.columns([4, 1])
                with col1:
                    movie_title = df_movies[df_movies['movieId'] == m_id]['title'].iloc[0]
                    st.write(f"• {movie_title}")
                with col2:
                    if st.button("❌", key=f"remove_{m_id}"):
                        st.session_state['preferred_movies_list'].remove(m_id)
                        st.rerun()
            
            preferred_movies = st.session_state['preferred_movies_list']
    
    # Số phim gợi ý
    k_recommendations = st.slider(
        "Số phim gợi ý:",
        min_value=5,
        max_value=20,
        value=10,
        step=1
    )
    
    # Threshold rating (chỉ cho collaborative filtering)
    if model_choice != 'Content-Based (Cold-Start)':
        threshold = st.slider(
            "Threshold rating (chỉ gợi ý phim >= ngưỡng này):",
            min_value=0.0,
            max_value=5.0,
            value=2.5,
            step=0.5
        )
    else:
        threshold = 0.0  # Content-based không cần threshold
    
    # Confidence threshold (chỉ cho ItemCF)
    confidence_threshold = 0.0
    if model_choice == 'ItemCF':
        confidence_threshold = st.slider(
            "Confidence threshold (ItemCF - chỉ gợi ý phim có confidence cao):",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Chọn 0.7+ để gợi ý chỉ những phim có confidence cao (🟢), 0.4+ cho trung bình (🟡 + 🟢)"
        )

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Metrics So Sánh Model")
    
    rmse_scores = metadata.get('rmse_scores') or metadata.get('rmse') or {}
    mae_scores = metadata.get('mae_scores') or metadata.get('mae') or {}
    precision_at_10 = metadata.get('precision_at_10') or {}
    recall_at_10 = metadata.get('recall_at_10') or {}

    metrics_data = {
        'Model': ['UserCF', 'ItemCF', 'SVD', 'Content-Based'],
        'RMSE': [
            rmse_scores.get('usercf', 'N/A'),
            rmse_scores.get('itemcf', 'N/A'),
            rmse_scores.get('svd', 'N/A'),
            rmse_scores.get('content', 'N/A')
        ],
        'MAE': [
            mae_scores.get('usercf', 'N/A'),
            mae_scores.get('itemcf', 'N/A'),
            mae_scores.get('svd', 'N/A'),
            mae_scores.get('content', 'N/A')
        ],
        'Precision@10': [
            precision_at_10.get('usercf', 'N/A'),
            precision_at_10.get('itemcf', 'N/A'),
            precision_at_10.get('svd', 'N/A'),
            precision_at_10.get('content', 'N/A')
        ],
        'Recall@10': [
            recall_at_10.get('usercf', 'N/A'),
            recall_at_10.get('itemcf', 'N/A'),
            recall_at_10.get('svd', 'N/A'),
            recall_at_10.get('content', 'N/A')
        ]
    }
    df_metrics = pd.DataFrame(metrics_data)
    st.dataframe(df_metrics, use_container_width=True)
    
    st.write("**Hướng dẫn chọn model:**")
    st.info(
        "• **ItemCF**: RMSE & MAE thấp nhất (0.9499, 0.7146) ➜ Dự đoán rating chính xác\n"
        "• **SVD**: Precision & Recall tốt nhất ➜ Gợi ý phim mới, tìm phim liên quan\n"
        "• **UserCF**: Hiệu năng trung bình ➜ Tìm người dùng có hành vi tương tự"
    )

with col2:
    st.subheader("Thông tin User")
    
    # Lấy thông tin user (df_train được load từ models/df_train.csv)
    user_ratings = df_train[df_train['userId'] == selected_user]
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Số Phim Đã Rate", len(user_ratings))
    with col_b:
        st.metric("Avg Rating", f"{user_ratings['rating'].mean():.2f}/5.0")
    with col_c:
        st.metric("Max Rating", f"{user_ratings['rating'].max():.1f}/5.0")
    
    st.write("**Một số phim đã đánh giá:**")
    sample_rated = user_ratings.sort_values('rating', ascending=False).head(5)
    
    for idx, (_, row) in enumerate(sample_rated.iterrows(), 1):
        movie_info = df_movies[df_movies['movieId'] == row['movieId']]
        if not movie_info.empty:
            movie_info = movie_info.iloc[0]
            st.write(f"{idx}. **{movie_info['title']}** ({movie_info.get('genres', 'N/A')}) — ⭐ {row['rating']}/5")

st.markdown("---")

# Gợi ý phim
st.subheader("Gợi Ý Phim")

if model_choice == 'Content-Based (Cold-Start)':
    # Content-based recommendation
    if not preferred_movies:
        st.warning("Vui lòng chọn ít nhất một phim mà bạn thích để nhận gợi ý.")
    else:
        st.write(f"**Gợi ý dựa trên nội dung (Content-Based):**")
        
        # Lấy danh sách phim đã rated của user (để exclude) - df_train được load từ models/df_train.csv
        user_rated_movies = set(df_train[df_train['userId'] == selected_user]['movieId'].values)
        
        df_recs = recommend_contentbased(
            preferred_movie_ids=preferred_movies,
            k=k_recommendations,
            exclude_movie_ids=user_rated_movies
        )
        
        if len(df_recs) > 0:
            st.dataframe(df_recs, use_container_width=True)
        else:
            st.warning("Không tìm thấy phim tương tự nào.")

elif model_choice == 'Compare All':
    # So sánh tất cả 5 model
    st.write(f"**Gợi ý cho User {selected_user} - So sánh 5 Model:**")
    
    for model_name in ['Hybrid', 'ItemCF', 'SVD', 'UserCF', 'Content']:
        st.markdown(f"### 🔹 Model: {model_name}")
        model_key = model_name.lower() if model_name != 'Content' else 'content'
        # Truyền confidence_threshold chỉ khi gọi ItemCF
        conf_threshold = confidence_threshold if model_name == 'ItemCF' else 0.0
        df_recs, error = recommend_movies(selected_user, k=k_recommendations, model_type=model_key, threshold=threshold, confidence_threshold=conf_threshold)
        
        if error:
            st.error(f"❌ {error}")
        elif df_recs is not None and len(df_recs) > 0:
            # ItemCF hiển thị confidence, các model khác hiển thị score
            if model_name.lower() == 'itemcf' and 'Confidence' in df_recs.columns:
                display_cols = ['Rank', 'Title', 'Predicted Rating', 'Confidence', 'Conf Score', 'Neighbors', 'Rating Range']
            else:
                display_cols = ['Rank', 'Title', 'Predicted Rating', 'Genres']
            display_cols = [c for c in display_cols if c in df_recs.columns]
            st.dataframe(df_recs[display_cols], use_container_width=True)
        else:
            st.warning("Không có gợi ý phù hợp với ngưỡng rating này")
        st.markdown("---")
else:
    # Chỉ một model
    st.write(f"**Gợi ý cho User {selected_user} - Model: {model_choice}**")
    
    df_recs, error = recommend_movies(selected_user, k=k_recommendations, model_type=model_choice.lower(), threshold=threshold, confidence_threshold=confidence_threshold)
    
    if error:
        st.error(f"❌ {error}")
    elif df_recs is not None and len(df_recs) > 0:
        # Hiển thị table với columns phù hợp
        if model_choice.lower() == 'itemcf' and 'Confidence' in df_recs.columns:
            # ItemCF: hiển thị confidence details
            display_cols = ['Rank', 'Title', 'Predicted Rating', 'Confidence', 'Conf Score', 'Neighbors', 'Rating Range']
            display_cols = [c for c in display_cols if c in df_recs.columns]
            st.dataframe(df_recs[display_cols], use_container_width=True)
            
            # Thêm hintlet
            st.info(
                "**📊 Confidence Details (ItemCF):**\n"
                "• 🟢 High (≥0.7): Dự đoán đáng tin cậy, neighbors đồng ý cao\n"
                "• 🟡 Medium (0.4-0.7): Dự đoán trung bình, có phân bổ\n"
                "• 🔴 Low (<0.4): Dự đoán không chắc chắn, neighbors không đồng ý"
            )
        else:
            # Các model khác
            display_cols = ['Rank', 'Title', 'Predicted Rating', 'Genres']
            display_cols = [c for c in display_cols if c in df_recs.columns]
            st.dataframe(df_recs[display_cols], use_container_width=True)
    else:
        st.warning("Không có gợi ý phù hợp với ngưỡng rating này")

st.markdown("---")
st.success("Ứng dụng đã sẵn sàng! Bắt đầu khám phá gợi ý phim ngay.")
