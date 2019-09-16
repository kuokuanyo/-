#讀取
data = read.csv('social_network_ads.csv')
data = data[, 3:5]

#將應變數轉換成因子(迴歸才會自動轉換)
data$Purchased = factor(data$Purchased, levels = c(0, 1))

#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split = TRUE)
test_set = subset(data, split = FALSE)

#特徵縮放
train_set[-3] = scale(train_set[-3])
test_set[-3] = scale(test_set[-3])

#建立模型
library(randomForest)
model = randomForest(x = train_set[,-3], y = train_set[,3], ntree = 10)

#預測
y_pred = predict(model, test_set[-3])

#查看正確率
cm = table(test_set[,3], y_pred)

#視覺化
#train_set
library(ElemStatLearn)
set = train_set
#建立網格
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
#網格
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#預測網格的y值
y_grid = predict(model, grid_set)
#畫圖
plot(set[, -3],
     main = 'Random Forest(train set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
#將點上色
points(set[, -3], pch = 21, bg = ifelse(set[, 3] == 1, 'blue', 'orange'))

#test_set
set = test_set
#建立網格
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
#網格
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#預測網格的y值
y_grid = predict(model, grid_set)
#畫圖
plot(set[, -3],
     main = 'Random Forest(test set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
#將點上色
points(set[, -3], pch = 21, bg = ifelse(set[, 3] == 1, 'blue', 'orange'))
