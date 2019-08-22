#數據前處理
#載入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#匯入資料
#df.values 矩陣
data =pd.read_csv('salary_data.csv') 
x = data.iloc[:, :-1].values #不要最後一欄，矩陣(30*1)
#reshape裡的變數為-1時的作用是自動計算
#reshape(-1, 1)意思為將行數固定為1，列數自動計算
"""或使用
x = data.iloc[:, 0].values 1*30矩陣
x = np.array(x).reshape(-1, 1) 30*1矩陣
"""
y = data.iloc[:, 1].values

#不需要分類及執行虛擬變數
#也不需要執行特徵縮放(回歸裡已有)
#訓練測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                    test_size = 1/3, random_state = 0)

#創建簡單線性迴歸
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#fit()訓練train data
regressor.fit(x_train, y_train)

#預測測試集結果
y_pred = regressor.predict(x_test)

#視覺化
#點圖plt.scatter()
plt.scatter(x_test, y_test, color = 'red') 
#迴歸線plt.plot()
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
#主題plt.title
plt.title('Salary VS Experience (test set)')
#x軸標籤plt.xlabel()
plt.xlabel('Years of Experience')
#y軸
plt.ylabel('Salary')
#畫出plt.show()
plt.show()













