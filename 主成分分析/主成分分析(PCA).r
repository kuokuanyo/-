#主成分分析(PCA)
#不須使用應變數(無監督)
#read
data = read.csv('Wine.csv')

#train、test
library(caTools)
split = sample.split(data$Customer_Segment, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#feature scaling
train_set[,-14] = scale(train_set[, -14])
test_set[, -14] = scale(test_set[, -14])

#apply PCA
library(caret)
library(e1071)
#pcaComp表示最後提取出的自變數數量
pca = preProcess(x = train_set[, -14], method = 'pca', pcaComp = 2)
train_set = predict(pca, train_set)
train_set = train_set[c(2, 3, 1)] #欄位更換
test_set = predict(pca, test_set)
test_set = test_set[c(2, 3, 1)] #欄位更換

#SVM model
model = svm(formula = Customer_Segment ~ .,
            data = train_set,
            type = 'C-classification',
            kernel = 'linear')

#predict
y_pred = predict(model, test_set[, -3])

#confusion matrix
cm = table(test_set[, 3], y_pred)

#visual(train_set)
library(ElemStatLearn)
set = train_set
#上下左右留白一個單位
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
#網格
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('PC1', 'PC2')
#預測網格
y_grid = predict(model, grid_set)
#畫點
plot(set[, -3],
     main = 'model', xlab('PC1'), ylab('PC2'),
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'Deepskyblue',
                          ifelse(y_grid == 1, 'springgreen3', 'tomato')))
#將點上色
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', 
                          ifelse(set[, 3] == 1, 'green4', 'red3')))

#test_set
library(ElemStatLearn)
set = test_set
#上下左右留白一個單位
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
#網格
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('PC1', 'PC2')
#預測網格
y_grid = predict(model, grid_set)
#畫點
plot(set[, -3],
     main = 'model', xlab('PC1'), ylab('PC2'),
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 2, 'Deepskyblue',
                                         ifelse(y_grid == 1, 'springgreen3', 'tomato')))
#將點上色
points(set, pch = 21, bg = ifelse(set[, 3] == 2, 'blue3', 
                                  ifelse(set[, 3] == 1, 'green4', 'red3')))
