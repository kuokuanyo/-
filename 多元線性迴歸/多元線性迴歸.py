#多元線性迴歸
#虛擬變數(dummy variables)一定要減少一欄的虛擬變數
#例如:假設有三個虛擬變數，只能使用兩欄
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#讀取資料並設置變數
data = pd.read_csv('50_Startups.csv')
x = data.iloc[:, :-1].values #DF.values:矩陣
y = data.iloc[:, 4].values 

#需要處理分類變數及虛擬變數
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder()
#fit_transform()訓練並操作轉換
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])

#虛擬變數
onehotencoder = OneHotEncoder(categorical_features = [3],
                              handle_unknown='ignore')
#直接使用fit_transform()執行訓練及轉換
x = onehotencoder.fit_transform(x).toarray()
#刪除三欄虛擬變數中其中的一欄(避免共線性)
#三個虛擬變數只能取兩欄
x = x[:, 1:] #刪除第一欄

#分測試訓練集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2, random_state = 0)

#線性迴歸不需要進行特徵縮放
#多元線性迴歸
from sklearn.linear_model import LinearRegression
#建立迴歸模型
regressor = LinearRegression()
#訓練迴歸模型
regressor.fit(x_train, y_train)

#預測結果
y_pred = regressor.predict(x_test)

#逐步迴歸(倒退消去)
import statsmodels.api as sm #不包含常數項(必須新增一欄)
#np.ones(40, 1) 建立40 * 1的矩陣(數值都為1)
#axis = 1 欄
x_train = np.append(arr = np.ones((40, 1)),
                    values = x_train, axis = 1)

#模型篩選
#先將所有欄位加入
x_opt = x_train[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

#刪除 x2欄(p-value = 0.85)
x_opt = np.delete(x_opt, 2, axis = 1)
#也可以使用 x_opt = x_train[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

#刪除 x1 欄
x_opt = np.delete(x_opt, 1, axis = 1)
#也可以使用 x_opt = x_train[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
regressor_OLS.summary()

#刪除 x2 欄
x_opt = np.delete(x_opt, 2, axis = 1)
#也可以使用 x_opt = x_train[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
print(regressor_OLS.summary()) #都顯著

#刪除 x2 欄
x_opt = np.delete(x_opt, 2, axis = 1)
#也可以使用 x_opt = x_train[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y_train, exog = x_opt).fit()
print(regressor_OLS.summary()) #都顯著
















