# 1. وارد کردن کتابخانه‌های موردنیاز
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 2. بارگذاری دیتاست
file_path = 'energy_data.csv'  # مسیر فایل داده‌ها
data = pd.read_csv(file_path)

# نمایش اطلاعات کلی درباره داده‌ها
print("اطلاعات داده‌ها:")
print(data.info())
print(data.head())

# 3. جداسازی ویژگی‌ها و اهداف
X = data[['Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']]  # ویژگی‌ها
y = data[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']]  # خروجی‌ها

# 4. نرمال‌سازی داده‌ها
X_normalized = (X - X.min()) / (X.max() - X.min())

# 5. تقسیم داده‌ها به مجموعه‌های آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# 6. ایجاد مدل رگرسیون خطی و آموزش آن
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# ایجاد مدل درخت تصمیم و آموزش آن
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)

# 7. پیش‌بینی خروجی‌ها برای داده‌های تست
y_pred_linear = linear_model.predict(X_test)
y_pred_tree = tree_model.predict(X_test)

# 8. ارزیابی مدل‌ها
print("\nارزیابی مدل رگرسیون خطی برای هر منطقه:")
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_linear[:, i])
    mse = mean_squared_error(y_test.iloc[:, i], y_pred_linear[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_linear[:, i])
    print(f"{col} (Linear Regression):")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.2f}")

print("\nارزیابی مدل درخت تصمیم برای هر منطقه:")
for i, col in enumerate(y.columns):
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred_tree[:, i])
    mse = mean_squared_error(y_test.iloc[:, i], y_pred_tree[:, i])
    r2 = r2_score(y_test.iloc[:, i], y_pred_tree[:, i])
    print(f"{col} (Decision Tree):")
    print(f"  MAE: {mae:.2f}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.2f}")

# 9. رسم نمودار مقایسه‌ای برای اولین خروجی (Zone 1) برای هر دو مدل
plt.figure(figsize=(12, 6))
plt.plot(y_test.iloc[:, 0].values, label='Actual (Zone 1)', color='blue')
plt.plot(y_pred_linear[:, 0], label='Predicted (Linear Regression)', color='red', linestyle='dashed')
plt.plot(y_pred_tree[:, 0], label='Predicted (Decision Tree)', color='green', linestyle='dotted')
plt.title('Actual vs Predicted - Zone 1 Power Consumption')
plt.xlabel('Sample Index')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()
# استخراج اهمیت ویژگی‌ها از مدل درخت تصمیم
feature_importance = tree_model.feature_importances_
for i, col in enumerate(X.columns):
    print(f"Feature: {col}, Importance: {feature_importance[i]:.2f}")


# نمایش ضرایب مدل
print("\nضرایب مدل رگرسیون خطی:")
for i, col in enumerate(X.columns):
    print(f"{col}: {linear_model.coef_[:, i]}")
