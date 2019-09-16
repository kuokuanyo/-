#套件
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#read
data = pd.read_csv('Ads_CTR_Optimisation.csv')

#計算信賴區間上界
import math
N = 10000 #總人數
d = 10 #廣告數
#總共有十則廣告，因此初始化10個元素為0的list
numbers_of_selections = [0] * 10 #廣告被播放次數
sums_of_rewards = [0] * 10 #廣告總共被點擊次數
ads_selected = [] #紀錄每個人被發放哪一個廣告
total_reward = 0

for n in range(0, N):
    #每一位人數開始時初始化上界值以及廣告i
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            #平均點擊次數
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            #log如為0，無意義
            delta_i = math.sqrt(3 / 2 * math.log(n+1) / numbers_of_selections[i])
            #上界
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    reward = data.iloc[n, ad]
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
            
#視覺化
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number ogf times each ad was selected')
plt.legend()
plt.show()            
            
            
            
            
            
            
            
            
            
            
            
            