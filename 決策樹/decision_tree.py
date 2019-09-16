#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#讀取
data = pd.read_csv('social_network_ads.csv')
x = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

#測試訓練
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.25, random_state = 0)

#特徵縮放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#建立模型
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
model.fit(x_train, y_train)

#預測
y_pred = model.predict(x_test)

#查看正確率
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#視覺化
#train_set
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
#建立網格
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1,
                    x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(x_set[:, 1].min() - 1,
                    x_set[:, 1].max() + 1, step = 0.01))
#分界線
plt.contourf(x1, x2, model.predict(np.array(
        [x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i), label = j)
plt.title('decision tree(train_set)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()

#test_set
x_set, y_set = x_test, y_test
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1,
                     x_set[:, 0].max() + 1, step = 0.01),
                    np.arange(x_set[:, 1].min() - 1,
                     x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(x1, x2, model.predict(np.array(
        [x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange','blue'))(i), label = j)
plt.title('decision tree(test_set)')
plt.xlabel('Age')
plt.ylabel('EstimatedSalary')
plt.legend()
plt.show()

























