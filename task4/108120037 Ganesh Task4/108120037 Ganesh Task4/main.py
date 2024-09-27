from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import preprocessing
from sklearn.svm import SVC
import pandas as pd
import pickle

iris=pd.read_csv('iris.csv')

X=iris.iloc[:,0:4]
Y=iris.iloc[:,-1]

label_encoder = preprocessing.LabelEncoder()
Y=label_encoder.fit_transform(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y)
lin_reg=LinearRegression()
log_reg=LogisticRegression()
svc_model=SVC()
knn_model= KNeighborsClassifier(n_neighbors=5)
lin_reg=lin_reg.fit(x_train,y_train)
log_reg=log_reg.fit(x_train,y_train)
svc_model=svc_model.fit(x_train,y_train)
knn_model= knn_model.fit(x_train,y_train)

pickle.dump(lin_reg,open('lin_model.pkl','wb'))
pickle.dump(log_reg,open('log_model.pkl','wb'))
pickle.dump(svc_model,open('svc_model.pkl','wb'))
pickle.dump(svc_model,open('knn_model.pkl','wb'))


