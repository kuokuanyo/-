#簡單線性迴歸
data = read.csv('Salary_data.csv')
attach(data)

#不需要分類變數
#訓練測試集
library(caTools)
set.seed(123)
split = sample.split(data$Salary, SplitRatio = 2/3)
training_set = subset(data, split == TRUE)
testing_set = subset(data, split == FALSE)

#迴歸(訓練集)
regressor = lm(Salary ~ YearsExperience, data = training_set)
summary(regressor)

#預測測試集
y_pred = predict(regressor, newdata = testing_set)

#視覺化
#install.packages('ggplot2')
library(ggplot2)
#訓練集
ggplot() + 
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary VS Experience') + 
  xlab('Years of Experience') +
  ylab('Salary')

#測試集
ggplot() + 
  geom_point(aes(x = testing_set$YearsExperience, y = testing_set$Salary),
             colour = 'red') + 
  geom_line(aes(x = training_set$YearsExperience, y = predict(regressor, newdata = training_set)),
            colour = 'blue') + 
  ggtitle('Salary VS Experience') + 
  xlab('Years of Experience') +
  ylab('Salary')
