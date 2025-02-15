import numpy as nm
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data_set = pd.read_csv("E:\\Downloads\\Concrete_Data_Integer.csv")

X = data_set.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values
Y = data_set.iloc[:, 8].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

st_X = StandardScaler()
X_train = st_X.fit_transform(X_train)
X_test = st_X.transform(X_test)

regressor1 = DecisionTreeRegressor(random_state=0)
regressor2 = KNeighborsRegressor(n_neighbors=3)
regressor3 = SVR(kernel='linear')

regressor1.fit(X_train, Y_train)
regressor2.fit(X_train, Y_train)
regressor3.fit(X_train, Y_train)


Y_pred1 = regressor1.predict(X_test)
Y_pred2 = regressor2.predict(X_test)
Y_pred3 = regressor3.predict(X_test)


mse1 = mean_squared_error(Y_test, Y_pred1)
mse2 = mean_squared_error(Y_test, Y_pred2)
mse3 = mean_squared_error(Y_test, Y_pred3)

r2_1 = r2_score(Y_test, Y_pred1)
r2_2 = r2_score(Y_test, Y_pred2)
r2_3 = r2_score(Y_test, Y_pred3)

print("MSE DT:", mse1)
print("MSE KNN:", mse2)
print("MSE SVR:", mse3)

print("R2 DT:", r2_1)
print("R2 KNN:", r2_2)
print("R2 SVR:", r2_3)


categories = ["DT", "KNN", "SVR"]
values = [mse1, mse2, mse3]
plt.bar(categories, values)
plt.xlabel('Regressors')
plt.ylabel('MSE')
plt.show()