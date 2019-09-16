#憭??扯艘甇?
#??鞈??
data = read.csv('50_Startups.csv')

#??????
data$State = factor(data$State,
                    levels = c('New York', 'California', 'Florida'),
                    labels = c(1, 2,3))

#???毀??葫閰?
library(caTools)
set.seed(123)
split = sample.split(data$Profit, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#銝?閬敺萇葬?
#撱箇?艘甇豢芋???
#銝?閬???霈?銝凋?甈????銵??迨瘝蝺批???
attach(data)
model = lm(Profit ~ ., data = train_set)
summary(model)

#??葫皜祈岫???
y_pred = predict(model, test_set)

#蝭拚霈(???瘜?)
#??State
train_set = train_set[, -4]
model = lm(Profit ~ ., data = train_set)
summary(model)

#??Administration甈?(蝚砌???)
train_set = train_set[, -2]
#??蝙?train_set$R.D.Spend = NULL ???摰???
model = lm(Profit ~ ., data = train_set)
summary(model)

#??Marketing.Spend甈?(蝚砌???)
train_set = train_set[, -2]
#??蝙?train_set$R.D.Spend = NULL ???摰???
model = lm(Profit ~ ., data = train_set)
summary(model)
