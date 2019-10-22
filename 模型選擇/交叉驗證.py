#交叉驗證
#套件
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

#read
data = pd.read_csv('Social_Network_Ads.csv')
x = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

#train、test set
x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = 0.25)

#feature scaling(x)
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test) #上面fit過了

#build model(kernel SVM)
model = SVC(kernel = 'rbf')
model.fit(x_train, y_train)

#predict
y_pred = model.predict(x_test)

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

#k次交叉驗證
#cv為交叉驗證次數
accuracies = cross_val_score(estimator = model,
                             X = x_train, y = y_train, cv = 10)
accuracies.mean() #91%

#網狀搜尋
#超參數
parameters = [{'C':[1, 10, 100, 1000],'kernel':['linear']},
               {'C':[1, 10, 100, 1000], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
               ]
grid_search = GridSearchCV(estimator = model,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
