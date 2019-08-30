#SVM
#read.csv
data = read.csv('social_network_ads.csv')
data = data[3:5]

#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split ==FALSE)

#特徵縮放
train_set[-3] = scale(train_set[-3])
test_set[-3] = scale(test_set[-3])

#建立SVM模型
attach(data)
library(e1071)
model = svm(Purchased ~ ., data = train_set,
            type = 'C-classification', kernel = 'linear')

#預測測試集
y_pred = predict(model, newdata = test_set[-3])

#混淆矩陣(看正確機率)
cm = table(test_set[,3], y_pred) #80%

#視覺化
library(ElemStatLearn)
set = train_set
#-1, +1 兩邊留白(多一個單位)
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
#畫所有網格
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#預測所有網格的值
y_grid = predict(model, newdata = grid_set)
#畫所有的點
plot(set[, -3], main = 'Classifier (train_set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#將點上色
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#測試集
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(model, newdata = grid_set)
plot(set[, -3], main = 'Classifier (test_set)',
     xlab = 'Age', ylab = 'EstimatedSalary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))