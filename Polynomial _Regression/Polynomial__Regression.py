import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

from sklearn.linear_model import LinearRegression
regressor_linear = LinearRegression()
regressor_linear.fit(x, y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg =  PolynomialFeatures(degree = 4)
MatrixOfFeatures_x_Poly = poly_reg.fit_transform(x)
regressor_linear2 = LinearRegression()
regressor_linear2.fit(MatrixOfFeatures_x_Poly, y)

#create a plot for linear regression for comparison
plt.scatter(x, y, color="red")
plt.plot(x, regressor_linear.predict(x), color="green")
plt.title("Truth or bluff (Linear Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show();

#create a plot for Polynomial regression
plt.scatter(x, y, color="red")
plt.plot(x, regressor_linear2.predict(MatrixOfFeatures_x_Poly), color="green")
plt.title("Truth or bluff (Polynomial Regression)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show();

#create a better lookin curves
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, regressor_linear2.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Truth or bluff (Polynomial Regression smoother)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show();

regressor_linear.predict([[6.5]]);
regressor_linear2.predict(poly_reg.fit_transform([[6.5]]));