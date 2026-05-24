# Movie Recommendation System - Trang Web Test Model

## Tất cả đã hoàn thành!

Bạn đã có:
- 3 model đã train (UserCF, ItemCF, SVD)
- Tất cả model đã được lưu trong folder `models/`
- Trang web tương tác để test model

---

## HDThực Hiện - Chạy Trang Web

### **Cách 1: Chạy từ PowerShell (Khuyến khích)**

1. **Mở PowerShell** trong folder project
2. **Chạy lệnh:**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   streamlit run app.py
   ```

3. **Tự động mở trình duyệt** với URL: `http://localhost:8501`

### **Cách 2: Chạy từ VS Code**

1. **Mở Terminal** (Ctrl + `)
2. **Paste lệnh:**
   ```powershell
   streamlit run app.py
   ```
3. **Nhấn Enter** → Trình duyệt sẽ tự động mở

---

## Hướng Dẫn Sử Dụng Trang Web

### **Giao Diện Chính**

```
┌─────────────────────────────────────────────────────────┐
│  🎬 Movie Recommendation System                          │
├──────────────┬──────────────────────────────────────────┤
│  SIDEBAR:    │  MAIN CONTENT:                           │
│              │                                          │
│ Chọn User ID │  Metrics So Sánh Model              │
│ Chọn Model   │  ├─ UserCF  | ItemCF  | SVD            │
│ Số phim gợi  │  └─ RMSE, MAE, Precision, Recall      │
│ Threshold    │                                          │
│              │  Thông tin User                      │
│              │  ├─ Số phim đã rate                     │
│              │  ├─ Avg Rating                          │
│              │  └─ Phim đã đánh giá                    │
│              │                                          │
│              │  Gợi Ý Phim                         │
│              │  └─ Bảng phim gợi ý (top-K)           │
│              │                                          │
│              │  Chi tiết phim (nếu chọn 1 model)  │
│              │     → Predicted rating, thể loại       │
└──────────────┴──────────────────────────────────────────┘
```

### **Các Bước Sử Dụng:**

**Step 1: Chọn User**
- Scroll dropdown "Chọn User ID"
- Chọn một user (VD: User 1, User 414, v.v.)
- App sẽ hiển thị số phim đã rate, avg rating, v.v.

**Step 2: Chọn Model**
- 4 lựa chọn:
  - ⭐ **ItemCF**: Dự đoán rating chính xác (MAE thấp)
  - **SVD**: Gợi ý phim chất lượng (Precision/Recall cao)
  - **UserCF**: Tìm user tương tự
  - **Compare All**: So sánh cả 3 model cạnh nhau

**Step 3: Điều chỉnh Tham Số**
- **Số phim gợi ý** (5-20): Bao nhiêu phim muốn thấy
- **Threshold rating** (0-5): Chỉ gợi ý phim nào > ngưỡng này

**Step 4: Xem Kết Quả**
- Nếu chọn 1 model: Click vào phim để xem chi tiết
- Nếu chọn "Compare All": Xem 3 cột so sánh

---

## Giải Thích Metrics

### **RMSE (Root Mean Square Error)**
- Độ lệch căn bậc hai
- **Thấp hơn = Tốt hơn**
- Ví dụ: RMSE 0.95 = Dự đoán sai ±0.95 sao

```
ItemCF RMSE: 0.9499 ⭐ (THẤP NHẤT)
UserCF RMSE: 1.0473
SVD RMSE:    3.3715 (CẢO NHẤT)
```

### **MAE (Mean Absolute Error)**
- Sai số tuyệt đối trung bình
- **Thấp hơn = Tốt hơn**
- Ví dụ: MAE 0.71 = Dự đoán sai ±0.71 sao trung bình

```
ItemCF MAE: 0.7146 ⭐ (THẤP NHẤT)
UserCF MAE: 0.8281
SVD MAE:    3.1753 (CẢO NHẤT)
```

### **Precision@10**
- % phim gợi ý mà user thực sự thích
- **Cao hơn = Tốt hơn**

```
SVD Precision@10: 0.0030 ⭐ (CAO NHẤT)
ItemCF Precision@10: 0.0002
UserCF Precision@10: 0.0000
```

### **Recall@10**
- % phim user thích được tìm thấy trong top-10 gợi ý
- **Cao hơn = Tốt hơn**

```
SVD Recall@10: 0.0295 ⭐ (CAO NHẤT)
ItemCF Recall@10: 0.0016
UserCF Recall@10: 0.0000
```

---

## Chọn Model Nào?

### **Muốn Dự Đoán Rating Chính Xác?**
👉 Dùng **ItemCF**
- MAE thấp nhất: 0.7146
- RMSE thấp nhất: 0.9499
- Phù hợp: Rating prediction tasks

### **Muốn Gợi Ý Phim Chất Lượng?**
👉 Dùng **SVD**
- Precision & Recall tốt nhất
- Phù hợp: Top-K recommendation tasks

### **Muốn Tìm User Tương Tự?**
👉 Dùng **UserCF**
- Tìm user có hành vi giống nhau
- Phù hợp: Social recommendation

### **Muốn So Sánh Tất Cả?**
👉 Chọn **Compare All**
- Xem 3 model cạnh nhau
- Chậm hơn (phải tính 3 model)

---

## 📁 Cấu Trúc File

```
project/
├── app.py                      ← Trang web Streamlit
├── HOW_TO_RUN_WEB.md          ← Hướng dẫn chi tiết
├── WEB_USAGE_GUIDE.md         ← File này
├── Data_Processing.ipynb       ← Xử lý dữ liệu
├── Train.ipynb                 ← Train model
├── Test_Model.ipynb            ← Test model
├── .venv/                       ← Virtual environment
├── models/
│   ├── usercf_model.pkl        ← UserCF model
│   ├── itemcf_model.pkl        ← ItemCF model ⭐ BEST
│   ├── svd_model.pkl           ← SVD model
│   ├── model_metadata.pkl      ← Metrics & mappings
│   └── movies_metadata_for_testing.csv
├── ratings.csv                 ← Dữ liệu gốc
├── movies.csv                  ← Thông tin phim
├── tags.csv                    ← Tags phim
└── (các file xử lý dữ liệu khác)
```

---

## Làm Cách Nào Để Dừng App?

1. **Nhấn `Ctrl + C`** trong terminal nơi chạy `streamlit run app.py`
2. Hoặc **Đóng cửa sổ trình duyệt** (app sẽ tự dừng sau 1-2 phút)

---

## Các Tính Năng Của Trang Web

### **1️⃣ Metrics Dashboard**
- So sánh RMSE/MAE/Precision/Recall của 3 model
- Dễ dàng so sánh hiệu năng

### **2️⃣ User Info Panel**
- Hiển thị số phim user đã rate
- Avg rating của user
- Max rating
- Một số phim đã đánh giá

### **3️⃣ Recommendation Engine**
- Gợi ý top-K phim cho user
- So sánh 3 model cạnh nhau
- Tùy chỉnh threshold rating

### **4️⃣ Movie Details**
- Xem chi tiết từng phim gợi ý
- Predicted rating
- Thể loại phim
- Movie ID

### **5️⃣ Interactive Controls**
- Selectbox: Chọn user & model
- Slider: Số phim & threshold
- Dataframe: Bảng kết quả tương tác

---

## Tips & Tricks

### **Tip 1: Tìm User Hoạt Động Nhất**
- User 414 đã rate 2,697 phim (hoạt động nhất)
- Lựa chọn tốt để test model

### **Tip 2: Thay Đổi Threshold**
- Nếu quá ít gợi ý: Hạ threshold (VD: 2.0 thay vì 2.5)
- Nếu quá nhiều gợi ý: Tăng threshold

### **Tip 3: So Sánh Model**
- Chọn "Compare All" để xem 3 model cạnh nhau
- Có thể mất 10-30 giây (tùy máy)

### **Tip 4: Kiểm Tra Kết Quả**
- So sánh predicted rating với actual rating (nếu user đã rate phim đó)
- Xem có hợp lý không

---

## Fix Lỗi Thường Gặp

### **Lỗi 1: "File not found"**
```
FileNotFoundError: models/usercf_model.pkl
```
**Giải pháp**: Chạy `Train.ipynb` từ đầu đến cuối để tạo các model

### **Lỗi 2: "streamlit command not found"**
```
streamlit : The term 'streamlit' is not recognized
```
**Giải pháp**: 
```powershell
pip install streamlit
# Hoặc dùng đường dẫn đầy đủ:
.\.venv\Scripts\streamlit.exe run app.py
```

### **Lỗi 3: App chạy chậm**
- **Nguyên nhân**: "Compare All" phải tính 3 model
- **Giải pháp**: Chọn 1 model thay vì "Compare All"

### **Lỗi 4: User không có gợi ý**
- **Nguyên nhân**: Threshold quá cao hoặc user đã rate tất cả phim
- **Giải pháp**: Hạ threshold hoặc chọn user khác

---

## Dữ Liệu Tập Huấn Luyện

```
Số Users:      610
Số Movies:     9,720
Số Ratings:    100,834
Sparsity:      98.30% (dữ liệu rất thưa)
Avg Rating/User: 165.3
Avg Rating/Movie: 10.4
```

---

## Kỹ Thuật Chi Tiết

### **UserCF (User-based CF)**
```
Công thức:
  sim(u1, u2) = cosine_similarity(rating_vector_u1, rating_vector_u2)
  pred(u, i) = Σ(sim(u, u') * rating(u', i)) / Σ(sim(u, u'))
  
Độ phức tạp: O(n_users²)
```

### **ItemCF (Item-based CF)**
```
Công thức:
  sim(i1, i2) = cosine_similarity(rating_vector_i1, rating_vector_i2)
  pred(u, i) = Σ(sim(i, i') * rating(u, i')) / Σ(sim(i, i'))
  
Độ phức tạp: O(n_items²)
Ưu điểm: Nhanh, ổn định với dữ liệu thưa
```

### **SVD (Matrix Factorization)**
```
Công thức:
  R ≈ U × Σ × V^T
  pred(u, i) = U[u]:rating·V[i]:rating
  
Độ phức tạp: O(min(n_users, n_items) * n_components)
Ưu điểm: Xử lý embedding, ranking tốt
n_components = 50 (tuning parameter)
```

---

## Kế Tiếp (Optional)

Nếu muốn cải thiện thêm:
1. **Hybrid Model**: UserCF + Content-based
2. **Deep Learning**: Neural Collaborative Filtering (NCF)
3. **Context-aware**: Gợi ý dựa trên thời gian, ngữ cảnh
4. **Explainability**: Giải thích tại sao gợi ý phim đó
5. **Real-time**: Cập nhật model liên tục

---

## Liên Hệ/Hỗ Trợ

Nếu gặp vấn đề:
1. Kiểm tra lại `models/` có file không
2. Thử cài lại Streamlit: `pip install --upgrade streamlit`
3. Chạy tất cả notebooks lại từ đầu
4. Xem chi tiết lỗi: `streamlit run app.py --logger.level=debug`

---

**Chúc bạn vui vẻ testing! 🎬🍿**
