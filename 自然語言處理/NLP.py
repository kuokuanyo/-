#NLP(自然語言處理)
#套件
import numpy as np
import pandas as pd

#read
#tsv檔以tab分隔
data = pd.read_csv('Restaurant_Reviews.tsv',
                   delimiter = '\t', quoting = 3)

#清理文字資料
import re
import nltk
nltk.download('stopwords') #用途:處理虛詞
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer #用途:處理詞根

#複迴圈
#定義一個空文集
corpus = []

#處理第i列
for i in range(0,1000):
    #只留英文單字
    #' '使東西被清除後，單字不連在一起
    review = re.sub('[^a-zA-Z]', ' ', data['Review'][i])
    
    #處理大小寫
    review = review.lower()

    #轉換成list
    review = review.split()

    #處理詞根，例如:love、loves、loved...等都來自於love單字
    #清理虛詞(沒有意義的單字)
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')] 

    #字串轉換
    review = ' '.join(review)
    
    corpus.append(review)
    
#建立稀疏矩陣
from sklearn.feature_extraction.text import CountVectorizer
#max_features: 清除出現次數很少的單字
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
#應變數
y = data.iloc[:, 1].values

#隨機森林樹
#訓練測試集
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#模型
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train, y_train)

#預測
y_pred = model.predict(x_test)

#混淆矩陣
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



