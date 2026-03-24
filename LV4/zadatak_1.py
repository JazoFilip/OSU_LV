import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,r2_score
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("LV3/data_C02_emission.csv")

features = ["Engine Size (L)", 
           "Cylinders", 
           "Fuel Consumption City (L/100km)", 
           "Fuel Consumption Hwy (L/100km)", 
           "Fuel Consumption Comb (L/100km)"]

X = data[features]
y = data["CO2 Emissions (g/km)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

plt.scatter(X_train["Engine Size (L)"], y_train,color="blue", label="Train")
plt.scatter(X_test["Engine Size (L)"], y_test, color="red", label ="Test")

plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.title("Odnos veličine motora i emisije CO2")
plt.legend() 
plt.show()


plt.subplot(1, 2, 1)
plt.hist(X_train["Engine Size (L)"], bins=20, color="blue")
plt.xlabel("Engine Size (L)")
plt.ylabel("Frequency")
plt.title("prije skaliranja - veličina motora")



sc = MinMaxScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

plt.subplot(1, 2, 2)
plt.hist(X_train_scaled[:, 3], bins=20, color="blue")
plt.xlabel("Engine Size (L)")
plt.ylabel("Frequency")
plt.title("nakon skaliranja - veličina motora")
plt.show()


linearModel = lm.LinearRegression()
linearModel.fit(X_train_scaled, y_train)

print("Theta (koeficijent):", linearModel.coef_)
print("Theta 0 (intercept):", linearModel.intercept_)


y_test_pred = linearModel.predict(X_test_scaled)

plt.scatter(y_test, y_test_pred, color="blue")
plt.xlabel("Stvarne vrijednosti")
plt.ylabel("Predviđene vrijednosti")
plt.title("Odnos stvarnih i predviđenih vrijednosti")
plt.show()


MSE = mean_squared_error(y_test, y_test_pred)
print("Mean Squared Error:", MSE)
RMSE = np.sqrt(MSE)
print("Root Mean Squared Error:", RMSE)
MAE = mean_absolute_error(y_test, y_test_pred)
print("Mean Absolute Error:", MAE)
MAPE = mean_absolute_percentage_error(y_test, y_test_pred)
print("Mean Absolute Percentage Error:", MAPE)
r2 = r2_score(y_test, y_test_pred)
print("R^2 Score:", r2)
