#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#讀取資料
data = pd.read_csv('social_network_ads.csv')
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

#訓練測試
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

#特徵縮放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#kernel SVM
from sklearn.svm import SVC
model = SVC(kernel = 'rbf', random_state = 0)
model.fit(x_train, y_train)

#預測測試集
y_pred = model.predict(x_test)

#查看正確機率
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#視覺化
#train test
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
#網格
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1,
                    x_set[:, 0].max() + 1, 0.01),
                    np.arange(x_set[:, 1].min() - 1,
                    x_set[:, 1].max() + 1,0.01))
#分界線
plt.contourf(x1, x2, model.predict(
        np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i),
                label = j)
    
plt.title('classifier train set')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()

#test test
from matplotlib.colors import ListedColormap
x_set, y_set = x_test, y_test
#網格
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1,
                    x_set[:, 0].max() + 1, 0.01),
                    np.arange(x_set[:, 1].min() - 1,
                    x_set[:, 1].max() + 1,0.01))
#分界線
plt.contourf(x1, x2, model.predict(
        np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i),
                label = j)
    
plt.title('classifier test set')
plt.xlabel('age')
plt.ylabel('estimated salary')
plt.legend()
plt.show()


















