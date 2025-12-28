# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Preparation: Load the California housing dataset, extract features (first three columns) and targets (target variable and sixth column), and split the data into training and testing sets.
2.Data Scaling: Standardize the feature and target data using StandardScaler to enhance model performance.
3.Model Training: Create a multi-output regression model with SGDRegressor and fit it to the training data.
4.Prediction and Evaluation: Predict values for the test set using the trained model, calculate the mean squared error, and print the predictions along with the squared error.


## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: vignesh s
RegisterNumber:  25014344
*/
```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print(df.head())


df.info()

X=df.drop(columns=['AveOccup','HousingPrice'])
X.info()

Y=df[['AveOccup','HousingPrice']]
Y.info()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)

print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)

Y_pred=multi_output_sgd.predict(X_test)
Y_pred=scaler_Y.inverse_transform(Y_pred)
Y_test=scaler_Y.inverse_transform(Y_test)
mse=mean_squared_error(Y_test,Y_pred)
print("Mean Squared Error:",mse)
print(Y_pred)

```

## Output:

<img width="735" height="285" alt="Screenshot 2025-11-21 140634" src="https://github.com/user-attachments/assets/be316ebb-205a-40a8-a730-b8626e94294d" />

<img width="414" height="353" alt="Screenshot 2025-11-21 140641" src="https://github.com/user-attachments/assets/e6ed9bce-162e-4b35-a40b-52be6eb904f8" />

<img width="415" height="311" alt="Screenshot 2025-11-21 140653" src="https://github.com/user-attachments/assets/bf11d40a-5124-4745-8c7f-d22cdc7456fb" />

<img width="439" height="205" alt="Screenshot 2025-11-21 140701" src="https://github.com/user-attachments/assets/3dc0d33d-936e-41af-81e3-a49423d1ae2a" />

<img width="606" height="834" alt="Screenshot 2025-11-21 140811" src="https://github.com/user-attachments/assets/db5eff30-f00d-45cb-a3f3-b9ba390e4f9f" />

<img width="257" height="129" alt="Screenshot 2025-11-21 140820" src="https://github.com/user-attachments/assets/c0eb8267-7cdf-4363-8bcd-3f9f36557bde" />

<img width="362" height="182" alt="Screenshot 2025-11-21 140826" src="https://github.com/user-attachments/assets/241dc510-6154-42ed-a823-9decf8c77cfb" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
