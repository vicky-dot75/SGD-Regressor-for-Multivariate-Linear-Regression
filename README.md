# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: vignesh s
RegisterNumber:  25014344
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()

X = data.data[:, [0, 1, 2, 5, 7]]  # Features: MedInc, HouseAge, AveRooms, AveOccup, Population
y = np.column_stack((data.target, data.data[:, 5]))  # Target: House Price, Occupancy

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sgd = SGDRegressor(max_iter=1000, tol=1e-3)
multi_output_sgd = MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train_scaled, y_train)

y_pred = multi_output_sgd.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

r2_house_price = r2_score(y_test[:, 0], y_pred[:, 0])
r2_occupancy = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"R² for House Price: {r2_house_price:.2f}")
print(f"R² for Occupancy: {r2_occupancy:.2f}")

*/


```

## Output:
<img width="735" height="285" alt="Screenshot 2025-11-21 140634" src="https://github.com/user-attachments/assets/f087d707-4663-488d-9bc2-46765d04845d" />

<img width="414" height="353" alt="Screenshot 2025-11-21 140641" src="https://github.com/user-attachments/assets/efa2edb9-d5e6-4068-9387-19814c0bb720" />

<img width="415" height="311" alt="Screenshot 2025-11-21 140653" src="https://github.com/user-attachments/assets/0f37f878-a784-4056-9cfd-f50ff12ed61e" />

<img width="439" height="205" alt="Screenshot 2025-11-21 140701" src="https://github.com/user-attachments/assets/47c41db4-80b6-43e6-9bd9-a615a0a4aec5" />

<img width="606" height="834" alt="Screenshot 2025-11-21 140811" src="https://github.com/user-attachments/assets/6a8b52e0-8097-4b01-a395-0447ab23c08e" />

<img width="257" height="129" alt="Screenshot 2025-11-21 140820" src="https://github.com/user-attachments/assets/8327ed03-3a0d-459c-aacd-a80afed845ee" />

<img width="362" height="182" alt="Screenshot 2025-11-21 140826" src="https://github.com/user-attachments/assets/94a5b2a0-7721-45d2-801c-028710b5c288" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
