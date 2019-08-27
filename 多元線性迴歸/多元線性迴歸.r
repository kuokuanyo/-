#多元線性迴歸
#匯入資料
data = read.csv('50_Startups.csv')

#分類資料
data$State = factor(data$State,
                    levels = c('New York', 'California', 'Florida'),
                    labels = c(1, 2,3))

#分訓練及測試
library(caTools)
set.seed(123)
split = sample.split(data$Profit, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#不需要特徵縮放
#建立迴歸模型
#不需要刪除虛擬變數其中一欄，會自動執行，因此沒共線性問題
attach(data)
model = lm(Profit ~ ., data = train_set)
summary(model)

#預測測試集
y_pred = predict(model, test_set)

#篩選變數(倒退刪除法)
#刪除State
train_set = train_set[, -4]
model = lm(Profit ~ ., data = train_set)
summary(model)

#刪除Administration欄(第二欄)
train_set = train_set[, -2]
#或使用train_set$R.D.Spend = NULL 刪除特定欄位
model = lm(Profit ~ ., data = train_set)
summary(model)

#刪除Marketing.Spend欄(第二欄)
train_set = train_set[, -2]
#或使用train_set$R.D.Spend = NULL 刪除特定欄位
model = lm(Profit ~ ., data = train_set)
summary(model)
