import pandas as pd #type:ignore
import numpy as np #type:ignore
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.linear_model import LinearRegression #type:ignore
import matplotlib.pyplot as plt #type:ignore

df = pd.read_csv("student_data.csv") #kaggle

# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.describe())
# print(df.info())
# correlation_matrix = df.select_dtypes(include=np.number).corr()
# print(correlation_matrix)

correlation = df.corr(numeric_only = True)
print(correlation["G3"].sort_values())

features = [ "Medu", "Fedu", "age", "goout", "G2", "G1", "failures"]

X = df[features]
Y = df["G3"] #final marks

X_train, X_test,Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state= 11)

model = LinearRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

table = {"Actual" : Y_test , "predicted" : Y_pred}
print(pd.DataFrame(table))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #type:ignore

mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
rmse = np.sqrt(mse)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

plt.scatter(range(len(Y_test)), Y_test, label = "Actual Grades", color = "red")
plt.scatter(range(len(Y_pred)), Y_pred, label = "Predicted Grades", color = "blue")
plt.xlabel("Actual Grade")
plt.ylabel("Predicted Grade")
plt.legend()
plt.grid(alpha = 0.4)
plt.savefig("graph.png")
plt.show()
