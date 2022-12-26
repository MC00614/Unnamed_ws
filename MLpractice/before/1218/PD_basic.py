import pandas as pd
import numpy as np

titanic_df = pd.read_csv('titanic_train.csv')
# print(titanic_df)
# print(titanic_df.head(3))
# print(titanic_df.shape)
# print(titanic_df.info())
# print(titanic_df.describe())
print(type(titanic_df['Pclass']==3))

print(titanic_df.isna().sum())