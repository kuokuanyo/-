#read
data = read.delim('Restaurant_Reviews.tsv', quote = '',
                  stringsAsFactors = FALSE)
#文本清理
library(tm)

#創建文集(處理第一欄的評論)
corpus = VCorpus(VectorSource(data$Review))

#處理大小寫(tm_map)
#content_transformer改變物件的型態
#先查看一開始corpus[[1]] ->"Wow... Loved this place."
#將所有轉換小寫(避免之後重複單字)
corpus = tm_map(corpus, content_transformer(tolower))
# "wow... loved this place."

#清除文字中的數字(與評論好壞無關)
corpus = tm_map(corpus, removeNumbers)

#清除標點符號(與評論好壞無關)
corpus = tm_map(corpus, removePunctuation)

#清除虛詞(單字本身無意義)
library(SnowballC) #使用stopwords()，處理虛詞
corpus = tm_map(corpus, removeWords, stopwords('en'))

#處理詞根，例如:love、loves、loved、loving...等詞根都是來自love
corpus = tm_map(corpus, stemDocument)

#處理空格
corpus = tm_map(corpus, stripWhitespace)

#建立一個稀疏矩陣
dtm = DocumentTermMatrix(corpus)
#清除出現次數過低的單字
#0.999只的是清除出現次數低於0.001的單字，例如總列數為1000，清除出現次數少於1的單字
dtm = removeSparseTerms(dtm, 0.999)

#預測
#利用隨機森林
#dtm資料轉換成data.frame並增加原始資料的Liked欄位
new_data = as.data.frame(as.matrix(dtm))
new_data$Liked = data$Liked
data = new_data

#因子轉換
data$Liked = factor(data$Liked, levels = c(0, 1))

#訓練測試集
library(caTools)
split = sample.split(data$Liked, SplitRatio = 0.8)
train_set = subset(data, split == TRUE)
test_set = subset(data, split == FALSE)

#不需要執行特徵縮放
#隨機森林
library(randomForest)
model = randomForest(x = train_set[, -692], y = train_set[, 692], ntree = 10)

#預測
y_pred = predict(model, test_set[, -692])

#混淆矩陣
cm = table(test_set[, 692], y_pred)

