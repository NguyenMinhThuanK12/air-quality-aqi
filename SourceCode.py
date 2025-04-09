# Import các thư viện cần thiết
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor  # Thêm mô hình XGBoost

# Bước 1: Hiểu dữ liệu và định nghĩa vấn đề
data = pd.read_csv('station_day.csv')

print("Thông tin dữ liệu ban đầu:")
print(data.info())
print("\nTổng số giá trị thiếu:") 
print(data.isnull().sum())
print("\nThống kê dữ liệu ban đầu:")
print(data.describe())

# Trực quan hóa phân phối dữ liệu ban đầu
plt.figure(figsize=(8, 6))
sns.histplot(data['AQI'].dropna(), bins=50, kde=True)
plt.title('Phân phối AQI ban đầu')
plt.xlabel('AQI')
plt.ylabel('Tần suất')
plt.show()

# Bước 2: Làm sạch và tiền xử lý dữ liệu
print("\nBắt đầu làm sạch và tiền xử lý dữ liệu...")
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')

# Loại bỏ cột không sử dụng
if 'AQI_Bucket' in data.columns:
    data = data.drop(columns=['AQI_Bucket'])

# Loại bỏ các dòng mà tất cả các chỉ số ô nhiễm đều trống
pollutant_columns = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']
data = data.dropna(subset=pollutant_columns, how='all')

print("\nDữ liệu sau khi loại bỏ các dòng trống tất cả các chỉ số ô nhiễm:")
print(data.info())

# Loại bỏ các cột có hơn 50% giá trị bị thiếu
missing_threshold = 0.5
col_threshold = missing_threshold * len(data)
data = data.dropna(thresh=col_threshold, axis=1)

print("\nDữ liệu sau khi loại bỏ các cột có hơn 50% giá trị bị thiếu:")
print(data.info())

# Điền giá trị thiếu
num_cols = data.select_dtypes(include=["float64"]).columns
num_imputer = SimpleImputer(strategy="mean")
data[num_cols] = num_imputer.fit_transform(data[num_cols])

print("\nDữ liệu sau khi điền giá trị thiếu (trung bình):")
print(data.info())

# Cập nhật danh sách pollutant_columns sau khi loại bỏ cột
pollutant_columns = [col for col in pollutant_columns if col in data.columns]

# Loại bỏ ngoại lệ bằng phương pháp clip theo percentiles
for col in pollutant_columns:
    upper_limit = data[col].quantile(0.99)
    lower_limit = data[col].quantile(0.01)
    data[col] = data[col].clip(lower=lower_limit, upper=upper_limit)

print("\nDữ liệu sau khi loại bỏ ngoại lệ:")
print(data.describe())

# Thêm cột "Season" dựa trên "Month"
data['Month'] = data['Date'].dt.month
data['Season'] = data['Month'].apply(lambda x: 'Spring' if x in [3, 4, 5] else
                                     'Summer' if x in [6, 7, 8] else
                                     'Autumn' if x in [9, 10, 11] else
                                     'Winter')

# Biến đổi "Season" thành dạng số
data['Season'] = data['Season'].map({'Spring': 1, 'Summer': 2, 'Autumn': 3, 'Winter': 4})

print("\nThông tin dữ liệu cuối cùng sau tiền xử lý:")
print(data.info())

# Trực quan hóa phân phối AQI theo mùa
plt.figure(figsize=(12, 6))
sns.boxplot(x='Season', y='AQI', data=data)
plt.title('Phân phối AQI theo mùa')
plt.xlabel('Mùa (Season)')
plt.ylabel('AQI')
plt.show()

# Heatmap tương quan
plt.figure(figsize=(12, 8))
corr_matrix = data[pollutant_columns].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Heatmap Tương Quan')
plt.show()




# Bước 4: Xây dựng mô hình dự đoán AQI
print("\nBắt đầu xây dựng và đánh giá mô hình dự đoán...")
features = data[["PM2.5", "PM10", "NO", "NO2", "NOx", "Season"]]
target = data['AQI']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Huấn luyện các mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=100),
    "XGBoost Regressor": XGBRegressor(random_state=42, n_estimators=100)
}

results = {}
for name, model in models.items():
    print(f"\nĐang huấn luyện mô hình: {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    results[name] = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    print(f"Kết quả mô hình {name}:")
    print(f"MAE: {mae}, RMSE: {rmse}, R2 Score: {r2}")

    # Trực quan hóa "Actual vs Predicted"
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Actual vs Predicted AQI ({name})')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.grid(True)
    plt.show()
