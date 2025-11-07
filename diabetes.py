import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

diabetes_data = pd.read_csv("Diabetes.csv")

#print(diabetes_data.groupby(' Class variable').mean())


X = diabetes_data.drop(columns=' Class variable', axis=1)
Y = diabetes_data[' Class variable']

# Data Standardization: Why it is required: Datas of the columns are in different ranges, it may cause the ML agorith some difficulty in computation. Thats why we standardize the data.
scaler = StandardScaler()

scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=2)

#print(X_train.shape, X_test.shape)

classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Evaluation
X_train_pred = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_pred, Y_train)
#print(training_data_accuracy)

X_test_pred = classifier.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_pred, Y_test)
#print(testing_data_accuracy)


# Predictive system

input_data = (1,103,30,38,83,43.3,0.183,33)
input_data_as_np_array = np.asarray(input_data)
input_data_reshape = input_data_as_np_array.reshape(1, -1)
input_standardize = scaler.transform(input_data_reshape)
print(input_standardize)

prediction = classifier.predict(input_standardize)

if prediction[0] == 'YES':
    print("Diabetic")
else:
    print("Not Diabetic")

