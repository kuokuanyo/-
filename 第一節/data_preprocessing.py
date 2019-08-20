#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#匯入資料
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values  #不取最後一欄（應變數）
y = dataset.iloc[:,3] #第三欄（應變數）

#利用平均值填補缺失值
from sklearn.preprocessing import Imputer #大寫為class
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0) #建立物件
imputer = imputer.fit(x[:, 1:3]) #小寫開頭為function
x[:, 1:3] = imputer.transform(x[:, 1:3]) #執行function

#分類資料
#使用虛擬編碼(dummy encoding)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder #大寫為class
labelencoder_x = LabelEncoder() #建立物件
x[:, 0] = labelencoder_x.fit_transform(x[:, 0]) #執行function
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#應變數只需要使用到LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#分成訓練及測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
 test_size = 0.2, random_state = 0)
