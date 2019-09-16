#read
data = read.csv('mall_customers.csv')
x = data[, 4:5]

#利用手肘法尋找最佳組數
set.seed(6) #將隨機數設為固定
wcss = vector()

#利用迴圈尋找10組裡的最佳組數
#tot.withinss 為組間距離
for (i in 1:10) wcss[i] = kmeans(x, i)$tot.withinss 

#畫圖
plot(1:10, wcss, type = 'b', main = 'The Elbow Method',
     xlab = 'Number of Clusters', ylab = 'WCSS')

#最佳組數為5
kmeans = kmeans(x, 5)
#分組訊息
y_kmeans = kmeans$cluster

#視覺化
library(cluster)
clusplot(x, y_kmeans, lines = 0, shade = TRUE, color = TRUE,
         labels = 2, plotchar = FALSE, span = TRUE, 
         main = paste('Clusters of customers'),
         xlab = 'Annual Income', ylab = 'Spending Score')