#SVM
#匯入套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read.csv
data = pd.read_csv('social_network_ads.csv')
x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

#分為測試訓練集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                        test_size = 0.25, random_state = 0)

#特徵縮放(應變數不需要，要0或1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#建立SVM模型
from sklearn.svm import SVC
model = SVC(kernel = 'linear', random_state = 0)
model.fit(x_train, y_train)

#預測
y_pred = model.predict(x_test)

#混淆矩陣(查看正確率) 88%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#視覺化
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train

#meshgrid畫出座標點上網格(可以為n維)
#產生所有的座標點
#-1 及+1 是要使兩邊留白(多一個單位)
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                    stop = x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,
                    stop = x_set[:, 1].max() + 1,step = 0.01))

#contour分界(預測網格的值並上色)
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(),
            x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#設定邊界
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
        c = ListedColormap(('orange','blue'))(i), label = j)

plt.title('classifier (train_set)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()    

#測試集
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1,
                    stop = x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = x_set[:, 1].min() - 1,
                    stop = x_set[:, 1].max() + 1,step = 0.01))

#contour分界(預測網格的值並上色)
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(),
            x2.ravel()]).T).reshape(x1.shape),
            alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#設定邊界
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
        c = ListedColormap(('orange','blue'))(i), label = j)

plt.title('classifier (test_set)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()    














