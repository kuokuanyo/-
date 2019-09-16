#read
data = read.csv('Ads_CTR_Optimisation.csv')

#初始化數值
#用戶數
n = 10000
#廣告數
d = 10
#每次被播放的廣告紀錄(空向量)
#integer(n)創造長度n的指定向量
ads_selected = integer()
#每個廣告獎勵為1或0的數值初始化
#十個都為0元素
numbers_of_reward_1 = integer(d)
numbers_of_reward_0 = integer(d)
#總獎勵數
total_reward = 0

for (n in 1:n) {
  #每個顧客都要初始化最大隨機數及選擇的廣告
  max_random = 0
  ad = 0
  for (i in 1:d) {
    #beta公式
    random_beta = rbeta(n = 1,
                        shape1 = numbers_of_reward_1[i] + 1,
                        shape2 = numbers_of_reward_0[i] + 1)
    #隨機變數樹否大於最大隨機變數
    if (random_beta > max_random) {
      max_random = random_beta
      ad = i
    }
  }
  #加入被選擇的廣告
  ads_selected = append(ads_selected, ad)
  #獎勵數值
  reward = data[n, ad]
  if (reward == 1) {
    numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1
  } else {
    numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
  }
  total_reward = total_reward + reward
}

#視覺化
hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')