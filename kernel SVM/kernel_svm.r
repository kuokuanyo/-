#kernel SVM
#read_csv
data = read.csv('social_network_ads.csv')
data = data[3:5]

#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split = TRUE)
test_set = subset(data, split = FALSE)

#特徵縮放(x)
train_set[-3] = scale(train_set[-3])
test_set[-3] = scale(test_set[-3])

#建立kernel SVM模型
#對train建立
library(e1071)
attach(data)
model = svm(Purchased ~ ., data = train_set,
            type = 'C-classification', kernel = 'radial')

#預測測試集結果
y_pred = predict(model, test_set[-3])

#查看正確率
cm = table(test_set[, 3], y_pred)

#視覺化
#train_set
library(ElemStatLearn)
set = train_set
#建立網格
#-1, +1 使兩邊留白
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
#預測網格的數值
y_grid = predict(model, newdata = grid_set)
#畫點
plot(set[, -3],
     main = 'kernel SVM (train set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
#水平線
contour(x1, x2, matrix(as.numeric(y_grid),length(x1), length(x2)), add = TRUE)
#將平分區域上色
points(grid_set, pch = '.', col = ifelse(y_grid != 1, 'red', 'green'))
#將點上色
points(set, pch = 21, bg = ifelse(set[, 3] != 1, 'orange', 'blue'))
  
#test_set
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
y_grid = predict(model, newdata = grid_set)
plot(set[, -3],
     main = 'kernel SVM (test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid != 1, 'red', 'green'))
points(set, pch = 21, bg = ifelse(set[, 3] != 1, 'orange', 'blue'))
  
  
  
  

  
  
  

  
  
  
  
  
  
  
  
  
  
  
  
  
  
      
  
  
  
  
  
  
  
  
  
  