#多項式迴歸
#讀取資料
data = read.csv("position_salaries.csv")
data = data[, 2:3]

#不需要測試訓練集(數據資料太少)
#不需要特徵縮放(迴歸不需)
#線性模型
attach(data)
model1 = lm(Salary ~ Level, data)
summary(model1)

#多項式迴歸
data$level2 = data$Level^2 #平方項
data$level3 = data$Level^3 #三次方項
data$level4 = data$Level^4 #四次方項
model2 = lm(Salary ~ ., data)
summary(model2)

#畫圖
library(ggplot2)

#線性迴歸
ggplot() + 
  geom_point(aes (x = Level, y = Salary),
             color = 'red') + 
  geom_line(aes (x = Level, y = predict (model1,data)),
            color = 'blue') +
  xlab('Level') +
  ylab('Salary')

#多項式迴歸
ggplot() + 
  geom_point(aes (x = Level, y = Salary),
             color = 'red') + 
  geom_line(aes (x = Level, y = predict (model2,data)),
            color = 'blue') +
  xlab('Level') +
  ylab('Salary')

#預測線性模型結果
y_pred1 = predict(model1, data.frame(Level = 6.5))

#預測多項式模型結果
y_pred2 = predict(model2, data.frame(Level = 6.5,
                                     level2 = 6.5 ^ 2,
                                     level3 = 6.5 ^ 3,
                                     level4 = 6.5 ^ 4))
