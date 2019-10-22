#主成分分析(PCA)
#無監督(不需要用到應變數)
#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#read
data = pd.read_csv('Wine.csv')
x = data.iloc[:, :13].values
y = data.iloc[:, 13].values

#train、test
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2)

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #前面fit過了

#apply PCA
pca = PCA(n_components = 2) #最後提取兩個自變數
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test) #前面fit過了
explained_variance = pca.explained_variance_ratio_

#邏輯迴歸
model = LogisticRegression()
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)

#confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#visual(train_set)
x_set, y_set = x_train, y_train
#上下左右都留白一格單位
#網格
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 1].max() + 1,
                               0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1,
                               0.01))
#分界線，上色
plt.contourf(x1, x2, model.predict(
        np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red', 'green', 'black')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#將點上色
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue', 'grey'))(i),
                label = j)

plt.title('Logistic Regression(train_set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#test_set
x_set, y_set = x_test, y_test
#上下左右都留白一格單位
#網格
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 1].max() + 1,
                               0.01),
                     np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1,
                               0.01))
#分界線，上色
plt.contourf(x1, x2, model.predict(
        np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
        alpha = 0.75, cmap = ListedColormap(('red', 'green', 'black')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())

#將點上色
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue', 'grey'))(i),
                label = j)

plt.title('Logistic Regression(test_set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
