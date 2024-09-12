
import pandas as pd
import joblib as jl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataframe = pd.read_csv("glass.csv")
x = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,9]

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20,random_state = 42)

scx = StandardScaler()
xtrain = scx.fit_transform(xtrain)
xtest = scx.transform(xtest)

rfc = RandomForestClassifier(criterion = "entropy",n_estimators = 300,random_state = 42)
rfc.fit(xtrain,ytrain)
ypredict = rfc.predict(xtest)

jl.dump(rfc,"classifier.sav")
print("ACCURACY is", rfc.score(xtest,ytest)*100,"%")