
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Загружаем данные
data = pd.read_csv(r"/home/user/PycharmProjects/WelcomeScreen/all_v2.csv", nrows=5000)

# Оставляем признаки, которые будем использовать
num_features = [
    "geo_lat",
    "geo_lon",
    "level",
    "levels",
    "rooms",
    "area",
    "kitchen_area",
]

categ_features = [
    "region",
    "building_type",
    "object_type"
]

# Оставляем нужные столбцы
data = data[features + ["price"]].copy()

# Убираем пропуски и бесконечные значения
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()

# Убираем явно некорректные значения
data = data[
    (data["price"] > 0) &
    (data["area"] > 0) &
    (data["kitchen_area"] >= 0) &
    (data["kitchen_area"] <= data["area"]) &
    (data["rooms"] >= 0) &
    (data["level"] >= 1) &
    (data["levels"] >= data["level"])
]

# Признаки и целевая переменная
X = data[features]
y = data["price"]

print("Количество объектов после очистки:", len(data))

# Разделяем данные: 80% на обучение, 20% на проверку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Создаём и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Получаем прогнозы
predictions = model.predict(X_test)

# Оцениваем качество
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("MAE:", round(mae, 2), "руб.")
print("RMSE:", round(rmse, 2), "руб.")

# Сравниваем настоящие и предсказанные цены
result = pd.DataFrame({
    "Настоящая цена": y_test.to_numpy(),
    "Предсказанная цена": map(int, predictions),
    "Ошибка" : (map(int, predictions - y_test.to_numpy()))
})
print("\nПримеры прогнозов:")
print(result.head(10))
