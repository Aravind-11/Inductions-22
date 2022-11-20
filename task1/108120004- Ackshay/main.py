import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
df=pd.read_table("Restaurant_Reviews.tsv")
print(df)
print(df.info())
print(df.describe())
print(df['Liked'].value_counts())

plt.figure(figsize=(8,5))
sns.countplot(x=df.Liked)

x=df['Review'].values
y=df['Liked'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
vect=CountVectorizer(stop_words='english')
x_train_vect=vect.fit_transform(x_train)
x_test_vect=vect.transform(x_test)
from sklearn.svm import SVC
model=SVC()
model.fit(x_train_vect,y_train)
y_pred=model.predict(x_test_vect)

Negative = df[df.Liked ==0]
Positive = df[df.Liked==1]
Negative_text = " ".join(Negative.Review.to_numpy().tolist())
Positive_text = " ".join(Positive.Review.to_numpy().tolist())

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  all_stopwords = stopwords.words('english')
  all_stopwords.remove('not')
  review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
  review = ' '.join(review)
  corpus.append(review)
print(corpus)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(corpus).toarray()
Y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
  
 ## checking model
                                                                      
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
print('Accuracy of NLP Support Vector Classifier Model is ', accuracy_score(Y_test, Y_pred))
print('\n','Confusion_Matrix:' '\n', confusion_matrix(Y_test,Y_pred))
print('\n', '\n','Report:' '\n',classification_report(Y_test,Y_pred))

from sklearn.pipeline import make_pipeline
text_model=make_pipeline(CountVectorizer(),SVC())
text_model.fit(x_train,y_train)
y_pred=text_model.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
print('Accuracy of NLP NB Model is ', accuracy_score(Y_test, Y_pred))
print('\n','Confusion_Matrix:' '\n', confusion_matrix(Y_test,Y_pred))
print('\n', '\n','Report:' '\n',classification_report(Y_test,Y_pred))

from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
pipeline = Pipeline([
    ('bow', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  
    ('classifier', MultinomialNB()),  
])

X = df['Review']
Y = df['Liked']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=0)


pipeline.fit(X_train,Y_train)
predictions = pipeline.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
print('Accuracy of NLP Pipeline Model is ', accuracy_score(Y_test, predictions))
print('\n','Confusion Matrix:' '\n', confusion_matrix(Y_test,predictions))
print('\n', '\n','Classification Report:' '\n',classification_report(Y_test,predictions))

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import  TfidfTransformer
pipeline_SVC = Pipeline([
    ('bow', CountVectorizer()),  
    ('tfidf', TfidfTransformer()),  #
    ('classifier', SVC()),  
])
pipeline_SVC.fit(X_train,Y_train)
predictions_SVC = pipeline_SVC.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score
print('Accuracy of NLP Pipeline Model with SVC is ', accuracy_score(Y_test, predictions_SVC))
print('\n','Confusion Matrix:' '\n', confusion_matrix(Y_test,predictions_SVC))
print('\n', '\n','Classification Report:' '\n',classification_report(Y_test,predictions_SVC))

import joblib
joblib.dump(pipeline_SVC,'Project')
import joblib
text_model=joblib.load('Project')
text_model.predict(['hello!!Love Your Food'])
text_model.predict(["omg!!it was too spice and i asked you don't add too much "])

import pickle 
pickle_out = open("classifier.pkl", mode = "wb") 
pickle.dump(text_model, pickle_out) 
pickle_out.close()
