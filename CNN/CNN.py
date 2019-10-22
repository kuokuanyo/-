#創建卷積神蹟網路
#套件
from keras.models import Sequential #初始化神經網路
from keras.layers import Convolution2D #創建卷積層
from keras.layers import MaxPooling2D #增加池化層
from keras.layers import Flatten #扁平層
from keras.layers import Dense #增加全連接層
from keras.preprocessing.image import ImageDataGenerator

#初始化
model = Sequential()

#卷積層
#filters:特徵探測器數量(32)
#特徵探測器為3*3矩陣
#第一個輸入層需要增加圖像大小單位(input_shape)
model.add(Convolution2D(32, (3, 3), activation = 'relu',
                        input_shape = (64, 64, 3)))

#池化層(目的是降維)
model.add(MaxPooling2D(pool_size = (2, 2)))

#優化性能
#增加第二個卷積層、池化層
model.add(Convolution2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

#扁平層
model.add(Flatten())

#全連接層(隱藏層)
model.add(Dense(units = 128, activation = 'relu'))
#輸出層(輸出結果為1個)
model.add(Dense(units = 1, activation = 'sigmoid'))

#編譯
#optimizer優化器
model.compile(optimizer = 'adam', loss = 'binary_crossentropy',
              metrics = ['accuracy'])

#圖像處理
train_datagen = ImageDataGenerator(rescale = 1./255, #縮放                        
                                   shear_range = 0.2, 
                                   zoom_range = 0.2, #放大縮小
                                   horizontal_flip = True) #水平翻轉

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode ='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit_generator(training_set,
                    steps_per_epoch = 250, 
                    epochs = 25, 
                    validation_data = test_set,
                    validation_steps = 62.5)


