import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

df = pd.read_csv('all_v2.csv', nrows=5_000)

print(df.describe())
print(f'пропуски: {df.isna().sum()}')
print(f'дубликаты: {df.duplicated().sum()}')
print(f'типы данных: {df.dtypes}')