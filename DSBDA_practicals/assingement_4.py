import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df=pd.read_csv("BostonHousing.csv")
print(df.head())
print(df.keys())
print(df.isnull().sum())

plt.figure(figsize=(10,10))
sns.heatmap(data=df.corr().round(2),annot=True,cmap='coolwarm',linewidth=0.2,square=True)
plt.show()

x = df.drop(columns=['medv'])     # storing all fratues in x except medv
y = df[ 'medv' ]                  # only medv feature is stored in y

x_train, x_test,y_train, y_test = train_test_split(x ,y, test_size=0.2, random_state=0)     #splitting data in 4 parts
# x_train = 80% of x data    |  x_test = 20% of y data
# y_train = 80% of x data    |  y_test = 20% of y data


lm = LinearRegression()      #only creating the learning model
lm.fit(x_train, y_train)     # learning from the data

y_train_pred = lm.predict(x_train)
y_test_pred = lm.predict(x_test)

mse_train =mean_squared_error(y_train,y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train MSE: {mse_train:.2f}, Train R²: {r2_train:.2f}")
print(f"Test MSE: {mse_test:.2f}, Test R²: {r2_test:.2f}")

plt.figure(figsize=(8, 6))

plt.scatter(y_train, y_train_pred, c='red', marker='o', label='Training Data')
plt.scatter(y_test, y_test_pred, c='green', marker='s', label='Test Data')

plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title("True vs Predicted House Prices")
plt.legend(loc='upper left')

plt.show()

coefficients = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
print(coefficients)

