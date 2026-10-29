import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('all_v2.csv', nrow=5000)

num_features = [
    "geo_lat",
    "geo_lon",
    "level",
    "levels",
    "rooms",
    "area",
    "kitchen_area"
]

cat_features = [
    "region",
    "building_type",
    "object_type"
]

features = num_features + cat_features

data = data[features + ["price"]].copy()
data = data.dropna()

x = data[features]
y = data["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(transformers=[("categoral", OneHotEncoder(), cat_features), ("numeric", "passthrough", num_features)])

model = Pipeline([("preprocessor", preprocessor), ("regression", LinearRegression())])

model.fit(x_train, y_train)
predictions = model.predict(x_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_absolute_error(y_test, predictions))

print("MAE: ", mae)
print("RMSE: ", rmse)

weights = pd.DataFrame({"Признак": model.named_steps["preprocessor"].get_feature_names_out(), "Вес": model.named_steps["regression"].coef_})
print(weights.to_string())




