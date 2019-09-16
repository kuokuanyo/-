#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read
data = pd.read_csv('Market_Basket_Optimisation.csv',
                   header = None)
transactions = []
for i in range(0, 7501):
    #串列生成式
    transactions.append([str(data.iloc[i, j]) for j in range(0, 20)])
#先驗演算
from apyori import apriori
rules = apriori(transactions, min_support = 0.003,
        min_confidence = 0.2, min_lift = 3, min_length = 2)

#視覺化
results = list(rules)
myresults = [list(x) for x in results] 