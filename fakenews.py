import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk

def stemming (article_content):
    if not isinstance(article_content, str):
        return ""
    stemmed_content = re.sub('[^a-zA-Z]', ' ', article_content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stemmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

nltk.download('stopwords')
#print(stopwords.words('english'))

# Data preprocessing
news_data = pd.read_csv("fakenews.csv", encoding='latin1', engine='python', on_bad_lines='skip')
#print(news_data.shape)
#print(news_data.columns)
news_data = news_data.fillna("empty")
#print(news_data.head(2))
#print(news_data.isnull().sum())

X = news_data.drop(columns='labels', axis=1)
Y = news_data['labels']

port_stemmer = PorterStemmer()


news_data['article_content'] = news_data['article_content'].apply(stemming)
#print(news_data['article_content'])

X = news_data['article_content'].values
Y = news_data['labels'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)

X = vectorizer.transform(X)
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=7)

model = LogisticRegression()
model.fit(X_train, Y_train)

X_train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred, Y_train)
#print(train_data_accuracy)

X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)
#print(test_data_accuracy)

X_new = X_test[5]
pred = model.predict(X_new)
print("Actual value: ", Y_test[5])
print("Predicted value: ", pred[0])

if pred[0] == 0:
    print("Real news")
else:
    print("Fake news")






