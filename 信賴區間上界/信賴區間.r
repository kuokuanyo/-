#read
data = read.csv('Ads_CTR_Optimisation.csv')

#信賴區間上界
#10個廣告
d = 10 
#10000個用戶
n = 10000
#初始化廣告播放次數(10個元素起始值為0)
numbers_of_selections = integer(d)
sums_of_rewards = integer(d)
#紀錄每個人被發放哪一個廣告
ads_selected = integer()
total_reward = 0

for (n in 1:n) {
  #每一個客戶開始會初始化
  max_upper_bound = 0 #最大信賴區間上界
  ad = 0 #第i個廣告
  for (i in 1:d) {
    if(numbers_of_selections[i] > 0) {
      #平均點擊數
      average_reward = sums_of_rewards[i] / numbers_of_selections[i]
      delta_i = sqrt(3 / 2 * log(n) / numbers_of_selections[i])
      #信賴區間上界
      upper_bound = average_reward + delta_i
    }
    else {
      upper_bound = 1e400
    }
    if (upper_bound > max_upper_bound) {
      max_upper_bound = upper_bound
      ad = i
    } 
  }
  ads_selected = append(ads_selected, ad)
  #被播放過後會加一次
  numbers_of_selections[ad] = numbers_of_selections[ad] + 1
  #查看是否被點擊
  reward = data[n, ad]
  #廣告點擊數增加
  sums_of_rewards[ad] = sums_of_rewards[ad] + reward
  #總點擊數
  total_reward = total_reward + reward
} 

#視覺化
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'ads',
     ylab = 'Numbers of times each ad was selected')