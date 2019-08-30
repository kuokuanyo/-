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

#模型
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = 'liblinear')
classifier.fit(x_train, y_train)

#預測結果
y_pred = classifier.predict(x_test)

#評估性能(比較預設跟實際的差別)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#畫出訓練集結果
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
#meshgrid是用兩個座標軸上的點在平面上畫網格(也可以是n維)
#min() - 1與max() + 1目的是使圖的邊緣留白(多一個單位)
#二維(年齡、平均薪水)
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                    stop = x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,
                    stop = x_set[:, 1].max() + 1, step = 0.01))
#畫等高線圖
#填充顏色(紅色及綠色)
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#設定值範圍
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
#迴圈(畫點圖)
#enumerate會產生兩個數值(第一個為序列數值，第二個維設定的參數)
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i), label = j)
plt.title('classifier (train_set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

#測試集
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                    stop = x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,
                    stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i), label = j)
plt.title('classifier (test_set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



















