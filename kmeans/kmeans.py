#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#讀取
#無監督學習(沒有應變數)
data = pd.read_csv('mall_customers.csv')
x = data.iloc[:, 3:5].values

#使用手肘法則尋找最佳分割的群數
from sklearn.cluster import KMeans
#wcss為組間距離
wcss = []
for i in range(1,11): #1~10組
    #n_clusters 集群數
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    #訓練
    kmeans.fit(x)
    #組間距離在inertia_裡
    wcss.append(kmeans.inertia_)
#畫圖
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

#最佳組數為5
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 0)
#訓練並預測
y_kmeans = kmeans.fit_predict(x)

#視覺化
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1],
    s = 100, c = 'red', label = 'careful')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1],
    s = 100, c = 'blue', label = 'standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
    s = 100, c = 'green', label = 'target')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1],
    s = 100, c = 'cyan', label = 'careless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1],
    s = 100, c = 'magenta', label = 'sensible')

#畫中心點
#KMeans裡內建cluster_centers_屬性
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s = 300, c = 'yellow', label = 'centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score ')
plt.legend()
plt.show()




















