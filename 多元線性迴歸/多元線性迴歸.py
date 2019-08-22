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
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer([('one_hot_encoder',
            OneHotEncoder(categories='auto'), [3])],remainder='passthrough')
x = np.array(transformer.fit_transform(x), dtype=np.float)

'''
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
'''


