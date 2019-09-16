#ANN(人工神經網路)
#read
data = read.csv('Churn_Modelling.csv')
data = data[, 4:14]

#應變數不須轉換成因子(以是0或1)
#自變數欄位轉換成分類數據(國家、性別)
data$Geography = factor(data$Geography,
                        levels = c('France', 'Spain', 'Germany'),
                        labels = c(1, 2, 3))
data$Gender = factor(data$Gender,
                     levels = c('Female', 'Male'),
                     labels = c(1, 2))

#測試訓練集
library(caTools)
split = sample.split(data$Exited, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#特徵縮放
train_set[, -c(2, 3, 11)] = scale(train_set[, -c(2, 3, 11)])
test_set[, -c(2, 3, 11)] = scale(test_set[, -c(2, 3, 11)])

#建立模型
library(h2o)
h2o.init(nthreads = -1)
model = h2o.deeplearning(y = 'Exited',
                         training_frame = as.h2o(train_set),
                         activation = 'Rectifier',
                         hidden = c(6, 6), #2層隱藏層，每層有六個向量
                         epochs = 100, #100期
                         train_samples_per_iteration = -2)

#預測
prob = h2o.predict(model, newdata = as.h2o(test_set[-11]))
y_pred = ifelse(prob > 0.5, 1, 0)
y_pred = as.vector(y_pred)

#混淆矩陣
cm = table(test_set[, 11], y_pred)
