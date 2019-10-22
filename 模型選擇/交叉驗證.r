#交叉驗證

#read
data = read.csv('Social_Network_Ads.csv')
data = data[, 3:5]

#train、test set
library(caTools)
split = sample.split(data$Purchased, SplitRatio = 0.75)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#feature scaling
train_set[, 1:2] = scale(train_set[, 1:2])
test_set[, -3] = scale(test_set[, -3])

#build model(kernel SVM)
library(e1071)
model = svm(Purchased ~ ., data = train_set,
            type = 'C-classification', kernel = 'radial')

#predict
y_pred = predict(model, test_set[, -3])

#confusion matrix
cm = table(test_set[, 3], y_pred)

#交叉驗證
#k為交叉驗證次數
library(caret)
folds = createFolds(train_set$Purchased, k = 10)
cv = lapply(folds, function(x) {
  train_fold = train_set[-x,] #將選取的列從train_set刪除
  test_fold = train_set[x,] #將選取的列放入test_fold
  model = svm(Purchased ~ .,
              data = train_fold,
              type = 'C-classification', kernel = 'radial')
  y_pred = predict(model, test_fold[, -3])
  cm = table(test_fold[, 3], y_pred)
  accuracy = (cm[1, 1] + cm[2, 2])/(cm[1, 1] + cm[1, 2] + cm[2, 1] + cm[2, 2])
  return(accuracy)

}
)

mean(as.numeric(cv))

#網狀搜尋
#超參數
model = train(form = Purchased ~ ., data = train_set, method = 'svmRadial')