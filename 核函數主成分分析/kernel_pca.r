#核函數主成分分析(kernel PCA)

#read
data = read.csv('Social_Network_Ads.csv')
data = data[, 3:5]

#train、test
library(caTools)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#feature scaling(col:1,2)
train_set[, 1:2] = scale(train_set[, 1:2])
test_set[, 1:2] = scale(test_set[, 1:2])

#apply kernel-PCA
#features為提取自變數個數
library(kernlab)
kpca = kpca(~., data = train_set[, -3], kernel = 'rbfdot', features = 2)
#predict結果為矩陣，必須轉換成dataframe
train_set_pca = as.data.frame(predict(kpca, train_set))
#增加應變數
train_set_pca$Purchased = train_set$Purchased
test_set_pca = as.data.frame(predict(kpca, test_set))
test_set_pca$Purchased = test_set$Purchased

#build Logistic regresssion
model = glm(Purchased ~ ., family = binomial, data = train_set_pca)

#predict
prob_pred = predict(model, type = 'response', newdata = test_set_pca[, -3])
#將機率轉換成0或1
y_pred = ifelse(prob_pred > 0.5, 1, 0)

#confusion matrix
cm = table(test_set_pca[, 3], y_pred)

#visual
#train_set
library(ElemStatLearn)
set = train_set_pca
#網格
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('V1', 'V2')
#predict grid_set
prob_set = predict(model, type = 'response', grid_set)
#將prob_set改成1或0
y_grid = ifelse(prob_set > 0.5, 1, 0)
#畫點
plot(set[, -3],
     main = 'train_set', xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將分界區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#將點上色
points(set[, c(1,2)], pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

#test_set
set = test_set_pca
#網格
x1 = seq(min(set[, 1]) - 1, max(set[, 1]) + 1, by = 0.0075)
x2 = seq(min(set[, 2]) - 1, max(set[, 2]) + 1, by = 0.0075)
grid_set = expand.grid(x1, x2)
colnames(grid_set) = c('V1', 'V2')
#predict grid_set
prob_set = predict(model, type = 'response', grid_set)
#將prob_set改成1或0
y_grid = ifelse(prob_set > 0.5, 1, 0)
#畫點
plot(set[, -3],
     main = 'train_set', xlab = 'V1', ylab = 'V2',
     xlim = range(x1), ylim = range(x2))
#分界線
contour(x1, x2, matrix(as.numeric(y_grid), length(x1), length(x2)), add = TRUE)
#將分界區域上色
points(grid_set, pch = '.', col = ifelse(y_grid == 1, 'springgreen3', 'tomato'))
#將點上色
points(set[, c(1,2)], pch = 21, bg = ifelse(set[, 3] == 1, 'green4', 'red3'))

