#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#匯入資料
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  #不取最後一欄（應變數）
y = dataset.iloc[:, 3] #第三欄（應變數）

#利用平均值填補缺失值
from sklearn.preprocessing import Imputer #大寫為class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #建立物件
#對資料做訓練fit()
imputer = imputer.fit(x[:, 1:3]) #小寫開頭為function
#將資料執行轉換的操作transform()
x[:, 1:3] = imputer.transform(x[:, 1:3]) #執行function

#分類資料
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #大寫為class
labelencoder_x = LabelEncoder() #建立物件
#直接使用fit_transform()執行訓練及轉換
x[:, 0] = labelencoder_x.fit_transform(x[:, 0]) #執行function

#建立虛擬編碼(無大小之分)
onehotencoder = OneHotEncoder(categorical_features = [0])
#直接使用fit_transform()執行訓練及轉換
x = onehotencoder.fit_transform(x).toarray()

#應變數只需要使用到LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#分成訓練及測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
 test_size = 0.2, random_state = 0)

#特徵縮放
#因年齡與薪水數值差異過大，如直接計算，值幾乎只被薪水影響，因此需要使用特徵縮放
#有兩種方法，標準化，正規化
#標準化是利用平均值做計算
#正規化是利用最大最小值計算
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler() #class
#對自變量特徵縮放
x_train = sc_x.fit_transform(x_train)
#因x_train已經fit()過了，因此x_test直接用transform()轉換
x_test = sc_x.transform(x_test)
#應變量是類別變量，不需要做特徵縮放
#假設在回歸中，應變數的數值非常大時，需要執行特徵縮放
