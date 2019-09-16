#湯普森隨機算法
#套件
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#初始化
import random
#總用戶
n = 10000
#總廣告
d = 10
#廣告獎勵1及0的總次數
#十個廣告各自對應不同數值，因此初始化十個元素
numbers_of_reward_1 = [0] * d
numbers_of_reward_0 = [0] * d
#每一次被選擇的廣告
ads_selected = []
total_reward = 0
for n in range(0, n):
    #每一個用戶計算時都要初始化被選擇的廣告及最大隨機數
    ad = 0
    max_random = 0
    for i in range(0, d):
        #beta公式
        random_beta = random.betavariate(numbers_of_reward_1[i] + 1,
                                         numbers_of_reward_0[i] + 1)
        #是否更改最大隨機數
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    #將用戶所背投放的廣告加入ads_selected
    ads_selected.append(ad)
    #取得廣告的獎勵數值
    reward = data.iloc[n, ad]
    #將獎勵數值加入 numbers_of_reward_1或 numbers_of_reward_0
    if reward == 1:
        numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
    else:
        numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
    total_reward = total_reward + reward
    
#視覺化
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.legend()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    