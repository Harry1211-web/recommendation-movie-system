# 🎬 Hướng Dẫn Chạy — MovieLens ml-25m

## Bước 0: Tải Dataset ml-25m

Tải từ: https://grouplens.org/datasets/movielens/25m/

Sau khi tải về, giải nén và đặt vào:
```
data/origin/ml-25m/
├── ratings.csv          # 25M rows
├── movies.csv
├── tags.csv
├── links.csv
├── genome-scores.csv
└── genome-tags.csv
```

## Bước 1: Cài Đặt Packages

```powershell
pip install pandas numpy scikit-learn scikit-surprise joblib scipy pyarrow requests streamlit tqdm matplotlib seaborn plotly
```

## Bước 2: Data Processing

Mở và chạy toàn bộ `Data_Processing.ipynb`:
- Load 6 file ml-25m
- Gọi TMDB API cho từng phim (có cache → tiếp tục được nếu dừng giữa chừng)
- Tính imdb_score (Bayesian Weighted Rating)
- Export ra `data/cleaning/*.parquet`

⏱️ Ước tính: 2–4 giờ (chủ yếu do TMDB API call cho ~60K phim)

## Bước 3: Data Visualization

Mở và chạy `data_visualization.ipynb`:
- 9 nhóm biểu đồ phân tích
- Kết quả lưu tại `data/cleaning/viz_*.png`

## Bước 4: Train Models

Mở và chạy toàn bộ `Train.ipynb`:

| Bước | Nội dung | Thời gian |
|------|----------|-----------|
| 3 | 70/15/15 Split | ~5 phút |
| 5 | SVD GridSearch (108 combos) | ~3-6 giờ |
| 6 | Sparse ItemCF | ~10 phút |
| 7 | Content-Based profiles | ~15 phút |
| 8 | Hybrid alpha search | ~20 phút |
| 9 | IMDb beta search | ~10 phút |
| 10 | Final evaluation | ~15 phút |
| 11 | Save models | ~5 phút |

⏱️ Tổng: ~5-8 giờ

## Bước 5: Test Models

Mở và chạy `Test_Model.ipynb` để kiểm tra kết quả.

## Bước 6: Chạy Web App

```powershell
streamlit run app.py
```

Mở trình duyệt: http://localhost:8501

---

## 📊 Cấu Trúc Gợi Ý

```
User có < 5 ratings?
    ├── Có → Content-Based (Cold-Start)
    └── Không → Hybrid(α×SVD + (1-α)×ItemCF) × (1 + β×imdb_norm) → Top-K
```

## 🤖 5 Chế Độ Gợi Ý Trong App

| Model | Mô tả |
|-------|-------|
| 🏆 Hybrid + IMDb Boost | SVD+ItemCF kết hợp với IMDb quality filter (mặc định) |
| ⚡ Hybrid SVD + ItemCF | Hybrid thuần không boost |
| 🔬 SVD | Matrix Factorization |
| 🔗 ItemCF | Item-based Collaborative Filtering (sparse) |
| 📄 Content-Based | Dựa trên đặc trưng phim (cold-start) |
| 📊 Compare All | So sánh tất cả 5 model |
