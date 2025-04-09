# 🌫️ Dự án Dự đoán Chỉ số Chất lượng Không khí (AQI)

Đây là đồ án phân tích và dự đoán chất lượng không khí sử dụng tập dữ liệu thực tế với hơn 108.000 bản ghi, bao gồm các chỉ số PM2.5, PM10, NO₂, AQI,... Dự án bao gồm các bước tiền xử lý dữ liệu, trực quan hóa, chọn đặc trưng và huấn luyện mô hình học máy.

## 📁 Cấu trúc dự án
- `station_day.csv`: Tập dữ liệu gốc thu thập từ Kaggle.
- `SourceCode.py`: Tập tin Python chính để xử lý, phân tích và xây dựng mô hình.
- `Report_PTDL.pdf`: Báo cáo dự án (bằng tiếng Việt).
  
## 🛠️ Công nghệ sử dụng
- Python, pandas, matplotlib, seaborn, scikit-learn, XGBoost

## 📊 Điểm nổi bật
- Làm sạch và xử lý hơn 100.000 dòng dữ liệu môi trường
- Tạo đặc trưng mới (ví dụ: "Mùa")
- Xây dựng mô hình: Hồi quy tuyến tính, Random Forest, XGBoost
- Mô hình tốt nhất: **XGBoost (R² = 0.83, RMSE = 46.18)**

## 🚀 Cách chạy
1. Cài thư viện: `pip install -r requirements.txt`
2. Chạy script: `python SourceCode.py`
