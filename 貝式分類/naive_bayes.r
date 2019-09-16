#read
data = read.csv('Social_Network_Ads.csv')
data = data[, 3:5]

#因貝式分類沒有將y自動轉換成因子
data$Purchased = factor(data$Purchased, levels = c(0, 1))

#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split = TRUE)
test_set = subset(data, split = FALSE)

#特徵縮放
train_set[, -3] = scale(train_set[, -3])
test_set[, -3] = scale(test_set[, -3])

#建立貝式分類模型
library(e1071)
model = naiveBayes(x = train_set[, -3],
                   y = train_set[, 3])

#預測測試集
y_pred = predict(model, train_set)

#查看正確率
cm = table(test_set[, 3], y_pred)

#視覺化
#train_set
library(ElemStatLearn)
set = train_set
#建立網格
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#預測網格值y
y_grid = predict(model, grid_set)
#畫訓練集的點
plot(set[, -3],
     main = 'Naive Bayes (train set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1),length(x2)), add = TRUE)
#分界區域填色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
#將點上色
points(set[, -3], pch = 21, bg = ifelse(set[, 3] == 1, 'blue', 'orange'))

#test_set
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(model, grid_set)
plot(set[, -3],
     main = 'Naive Bayes (test set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
points(set[, -3], pch = 21, bg = ifelse(set[, 3] == 1, 'blue', 'orange'))
