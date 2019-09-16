#read
#利用read.transactions將所有產品整理
library(arules)
data = read.transactions('Market_Basket_Optimisation.csv',
                         sep = ',', rm.duplicates = TRUE)
#查看購買資訊
summary(data)
#畫出前n名的產品
itemFrequencyPlot(data, topN = 10)

#利用先驗演算法運算
#parameter設定支持度及信心度
rules = apriori(data, parameter = list(support = 0.003, confidence = 0.2))

#視覺化(以提升度排序)
inspect(sort(rules, by = 'lift')[1:10])