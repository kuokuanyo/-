#核函數主成分分析(kernel PCA)
#用於線性不可分數據
#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

#read
data = pd.read_csv('Social_Network_Ads.csv')
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

#train、test
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.2)

#feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #上面fit過了

#apply kernel-PCA 
kpca = KernelPCA(n_components = 2) #n_components為提取的自變數數量
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test) #上面fit過了

#Logistic regression
model = LogisticRegression()
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

#visual(train_set)
x_set, y_set = x_train, y_train
#網格(上下左右留白一個單位)
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1,
                               step = 0.01),
                    np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1,
                              step = 0.01))
#分界線
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
        x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
#將點上色
#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j, s = 5)
plt.title('Logistic Regression')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

#test_set
x_set, y_set = x_test, y_test
#網格(上下左右留白一個單位)
x1, x2 = np.meshgrid(np.arange(x_set[:, 0].min() - 1, x_set[:, 0].max() + 1,
                               step = 0.01),
                    np.arange(x_set[:, 1].min() - 1, x_set[:, 1].max() + 1,
                              step = 0.01))
#分界線
plt.contourf(x1, x2, model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(
        x1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(x1.min(), x1.max())
plt.ylim(x2.min(), x2.max())
#將點上色
#迴圈
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j, s = 5)
plt.title('Logistic Regression')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
