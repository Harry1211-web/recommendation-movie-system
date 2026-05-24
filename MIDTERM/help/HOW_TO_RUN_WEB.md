# Hướng Dẫn Chạy Trang Web Test Model

## Yêu cầu

Trước khi chạy, bạn cần:
1. Đã train xong các model (chạy `Train.ipynb`)
2. Các file model đã được lưu trong folder `models/`
3. Cài đặt Streamlit

## Bước 1: Cài đặt Streamlit

Streamlit là một framework Python để tạo web app tương tác mà không cần HTML/CSS/JavaScript phức tạp.

Mở **PowerShell** hoặc **Terminal** và chạy:

# Cài đặt Streamlit
pip install streamlit
```

## Bước 2: Chạy Ứng dụng Web

Trong cùng thư mục (một trong những folder chứa `app.py`), chạy:

```powershell
streamlit run app.py
```

Streamlit sẽ tự động mở trình duyệt và hiển thị ứng dụng tại `http://localhost:8501`

## Bước 3: Sử dụng Ứng dụng

### Trên trang web:

1. **Sidebar bên trái:**
   - **Chọn User ID**: Chọn user muốn nhận gợi ý (từ 1-610)
   - **Chọn Model**: ItemCF / SVD / UserCF / Compare All
   - **Số phim gợi ý**: Từ 5-20 phim
   - **Threshold rating**: Chỉ gợi ý phim có rating ≥ ngưỡng

2. **Phần trên (Metrics):**
   - So sánh RMSE, MAE, Precision, Recall của 3 model
   - Hướng dẫn chọn model phù hợp

3. **Phần gợi ý:**
   - Xem gợi ý phim cho user đã chọn
   - Click vào phim để xem chi tiết

## Cách Chọn Model

| Model | RMSE | MAE | Precision | Recall | Dùng cho |
|-------|------|-----|-----------|--------|----------|
| **UserCF** | 1.0473 | 0.8281 | 0.0000 | 0.0000 | Tìm user tương tự |
| **ItemCF** ⭐ | **0.9499** | **0.7146** | 0.0002 | 0.0016 | Dự đoán rating chính xác |
| **SVD** | 3.3715 | 3.1753 | **0.0030** | **0.0295** | Gợi ý phim mới |
| **Content-Based** | 1.7201 | 1.3214 | 0.0000 | 0.0000 | Cold-start, gợi ý theo nội dung |

### Gợi ý:
- Muốn **dự đoán rating chính xác** → dùng **ItemCF** (MAE thấp nhất)
- Muốn **gợi ý phim chất lượng** → dùng **SVD** (Precision/Recall tốt)
- Muốn so sánh tất cả → chọn **Compare All**

## Để dừng ứng dụng

Nhấn `Ctrl + C` trong terminal nơi bạn chạy `streamlit run app.py`

## Troubleshooting

### Lỗi: "File not found: models/..."
- **Giải pháp**: Đảm bảo bạn đã chạy `Train.ipynb` hoàn toàn để tạo folder `models/`

### Lỗi: "streamlit: command not found"
- **Giải pháp**: Chạy `pip install streamlit` lại hoặc kích hoạt `.venv` đúng cách

### Ứng dụng chạy chậm
- **Giải pháp**: Hạ số lượng user hoặc dùng `Compare All` sẽ chậm hơn do phải tính 3 model

## Cấu trúc file cần có

```
project-folder/
├── app.py                              # File web app
├── Train.ipynb                          # Notebook train model
├── Test_Model.ipynb                     # Notebook test
├── Data_Processing.ipynb                # Notebook xử lý dữ liệu
├── .venv/                               # Virtual environment
└── models/
    ├── usercf_model.pkl                # Model UserCF
    ├── itemcf_model.pkl                # Model ItemCF
    ├── svd_model.pkl                   # Model SVD
    ├── model_metadata.pkl              # Metadata (metrics, mappings)
    └── movies_metadata_for_testing.csv # Thông tin phim
```

## Chi tiết các Model

### UserCF (User-based Collaborative Filtering)
```
Công thức: rating_pred = Σ(similarity(user, similar_user) * rating_by_similar_user) / Σ(similarity)

Ưu điểm:
- Dễ hiểu, dễ implement
- Tìm user có hành vi tương tự

Nhược điểm:
- RMSE cao (1.0473)
- Kém hiệu quả với dữ liệu thưa
```

### ItemCF (Item-based Collaborative Filtering)
```
Công thức: rating_pred = Σ(similarity(item, rated_item) * rating_of_rated_item) / Σ(similarity)

Ưu điểm:
- RMSE & MAE thấp nhất (0.9499, 0.7146)
- Ổn định, dễ cải thiện
- Nhanh khi dữ liệu thưa

Nhược điểm:
- Precision/Recall thấp (chỉ tốt cho dự đoán, không gợi ý)
```

### SVD (Matrix Factorization)
```
Công thức: R ≈ U × Σ × V^T
           rating_pred = U[user] · V[item]

Ưu điểm:
- Pembelajaran embedding ẩn tốt
- Precision & Recall tốt nhất
- Xử lý sparsity tốt

Nhược điểm:
- RMSE cao (3.3715)
- Cần tuning tham số
```

## Cần giúp?

Nếu gặp vấn đề:
1. Kiểm tra lại file `models/` có đầy đủ chưa
2. Đảm bảo Streamlit được cài đặt: `pip list | grep streamlit`
3. Thử chạy lại: `streamlit run app.py --logger.level=debug`

---

**Chúc bạn vui vẻ khám phá hệ thống gợi ý phim! 🎬**
