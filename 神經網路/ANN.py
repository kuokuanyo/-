#人工神經網路(ANN)
#套件
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential #初始化神經網路
from keras.layers import Dense #處理層
from sklearn.metrics import confusion_matrix

#前處理
#read
data = pd.read_csv('Churn_Modelling.csv')
x = data.iloc[:, 3:13].values #只顯示值，不會顯示欄位
y = data.iloc[:, 13].values

#將國家(1)、性別(2)欄位數值分類
#國家
labelencoder_x_1 = LabelEncoder()
x[:, 1] = labelencoder_x_1.fit_transform(x[:, 1])
#性別
labelencoder_x_2 = LabelEncoder()
x[:, 2] = labelencoder_x_2.fit_transform(x[:, 2])

#分類變數轉虛擬變量
#性別只有兩個分類，不需要設為虛擬變數
#國家需要轉成虛擬變數
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()

#避免共線性，三個虛擬變數只能取兩欄
#取除第0欄
x = x[:, 1:]

#訓練測試集
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                test_size = 0.2)

#特徵縮放
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

#神經網路模型設定
#初始化神經網路
model = Sequential()

#增加輸入層及第一個隱藏層
model.add(Dense(units = 6, activation = 'relu',
                kernel_initializer = 'uniform',
                input_dim = 11))

#增加第二個隱藏層(不需要輸入層維度參數input_dim)
model.add(Dense(units = 6, activation = 'relu',
                kernel_initializer = 'uniform'))

#增加輸出層
#此數據輸出結果只有0或1因此設定sigmoid
#假設輸出結果大於兩種以上，activation = 'softmax'，units也需要更改
model.add(Dense(units = 1, activation = 'sigmoid',
                kernel_initializer = 'uniform'))

#編譯神經網路
#loss損失函數
#應變數為0或1的話，binary_crossentropy
#應變數大於三，categorical_crossentropy
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#訓練
#每10個數據重複做一次訓練
#重複100期(epochs)的訓練
model.fit(x_train, y_train, batch_size = 10, epochs = 100)

#預測測試集
y_pred = model.predict(x_test)
#大於0.5為1，反之0
y_pred = np.where(y_pred > 0.5, 1, 0)

#混淆矩陣
cm = confusion_matrix(y_test, y_pred)





