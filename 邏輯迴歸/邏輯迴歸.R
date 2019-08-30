#讀取資料
data = read.csv('social_network_ads.csv')
data = data[, 3:5]

#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#特徵縮放(對1, 2欄)
train_set[, 1:2] = scale(train_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

#建立模型
attach(data)
model = glm(Purchased ~ .,data = train_set, family = binomial)
summary(model)

#預測
prod_pred = predict(model, type = 'response', test_set[-3])
y_pred = ifelse(prod_pred > 0.5, 1, 0)

#製作confusion matrix
cm = table(test_set[, 3], y_pred)

#畫圖
library(ElemStatLearn)
#設置set方便以後要畫測試及時只需改一個值
set = train_set
#間隔越小，系統會跑越久
#-1及+1是使兩個邊界留白(多留一個單位)
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
#找出所有可能性列表(網格)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(model, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
#畫出所有集的值
plot(set[, -3],
     main = 'Classifier (Training set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
#畫出分界
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將網格區域畫出
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
#將點上色
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'orange', 'blue'))

#測試集
set = test_set
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.05)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.05)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('Age', 'EstimatedSalary')
prob_set = predict(model, type = 'response', newdata = grid_set)
y_grid = ifelse(prob_set > 0.5, 1, 0)
plot(set[, -3],
     main = 'Classifier (test set)',
     xlab = 'Age', ylab = 'Estimated Salary',
     xlim = range(x1), ylim = range(x2))
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'green', 'red'))
points(set, pch = 21, bg = ifelse(set[, 3] == 1, 'orange', 'blue'))
