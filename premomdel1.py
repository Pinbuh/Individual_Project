
import pandas as pd
import matplotlib.pyplot as plt

# Загружаем данные
data = pd.read_csv(r"C:\Users\Maibenben\PyCharmProject++\all_v2.csv", nrows=5000)

# Проверяем основные характеристики
print("Размер таблицы:", data.shape)
print("\nСтатистика цены:")
print(data["price"].describe())

# Проверяем подозрительные значения
print("\nСамые дорогие объекты:")
print(data.nlargest(10, "price")[
    ["price", "area", "rooms", "region"]
])

print("\nПлощади объектов:")
print(data["area"].describe())

# Строим гистограмму цены
data["price"].hist(bins=50)
plt.xlabel("Цена, руб.")
plt.ylabel("Количество объектов")
plt.title("Распределение цен недвижимости")
plt.show()
