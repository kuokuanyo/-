#多項式迴歸
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_csv('Position_salaries.csv')
x = data.iloc[:, 1:2].values #需要是矩陣(不然後面會有問題)
y = data.iloc[:, 2].values

#只有十個數據，不需要分成訓練測試集
#迴歸不需要特徵縮放
#建立線性模型與多項式迴歸模型

#線性
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(x, y)

#多項式
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) #n次多項式
x_poly = poly_reg.fit_transform(x) #常數項、一次項、二次項
model2 = LinearRegression()
model2.fit(x_poly, y)

#兩個迴歸模形圖
#線性
plt.scatter(x, y, color = 'red')
plt.plot(x, model1.predict(x), color = 'blue')
plt.title('Truth or  Bluff(linear regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#多項式迴歸
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1) #或者使用reshape(-1, 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, model2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')
plt.title('Truth or  Bluff(Polynomial regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()











