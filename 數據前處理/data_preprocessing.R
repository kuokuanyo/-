#匯入資料
dataset = read.csv('Data.csv')

#處理缺失值
#is.na：是否為na
dataset$Age[is.na(dataset$Age)] = mean(dataset$Age, na.rm = T) #na.rm：將na去除
dataset$Salary[is.na(dataset$Salary)] = mean(dataset$Salary, na.rm = T) #na.rm：將na去除

#分類資料
#使用factor轉換
dataset$Country = factor(dataset$Country)
dataset$Purchased = factor(dataset$Purchased)

#分為測試及訓練集
#install.packages('caTools')
library(caTools)
#set.seed():產生隨機數（讓之後產生的數值都一樣）
set.seed(123)
#sample.split():將數據分成訓練及測試集
#第一個變數為應變數(y)
split = sample.split(dataset$Purchased, SplitRatio = 0.8) #True為訓練集，False為測試集
training_set = subset(dataset, split == TRUE) #8筆
testing_set = subset(dataset, split == FALSE) #2筆

#特徵縮放
#scale()必須都是數值
#只對年齡和薪水執行特徵縮放
training_set[, 2:3] = scale(training_set[, 2:3])
testing_set[, 2:3] = scale(testing_set[, 2:3])
