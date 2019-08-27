#匯入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#讀取
data = pd.read_csv('social_network_ads.csv') 
x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

#訓練測式集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                test_size = 0.25, random_state = 0)
#y只能是0,1，不能夠特徵縮放
#對x做特徵縮放
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)
