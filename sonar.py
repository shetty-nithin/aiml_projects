"""
Logistic Logistic

Logistic Regression is used for classification problems where the output is categorical, usually binary(0 or 1, Yes or No, True or False)

Unlike Linear Regression, which predicts the ocntinuous values, logistic Regression predicts probabilities and maps then to a class label.

The core of logistic regression is the sigmoid function

Loss Function: We cannot use MSE like linear regression. Logistic Regression uses cross-entropy(logg loss)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('sonar.all-data.csv', header=None)

# Seprating data and label
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]

# Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# model training
model = LogisticRegression()
model.fit(X_train, Y_train)

# model evaluation
X_train_pred = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_pred, Y_train)
print(train_data_accuracy)

X_test_pred = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_pred, Y_test)
print(test_data_accuracy)

# making the predictive system
input_data = (0.0283,0.0599,0.0656,0.0229,0.0839,0.1673,0.1154,0.1098,0.1370,0.1767,0.1995,0.2869,0.3275,0.3769,0.4169,0.5036,0.6180,0.8025,0.9333,0.9399,0.9275,0.9450,0.8328,0.7773,0.7007,0.6154,0.5810,0.4454,0.3707,0.2891,0.2185,0.1711,0.3578,0.3947,0.2867,0.2401,0.3619,0.3314,0.3763,0.4767,0.4059,0.3661,0.2320,0.1450,0.1017,0.1111,0.0655,0.0271,0.0244,0.0179,0.0109,0.0147,0.0170,0.0158,0.0046,0.0073,0.0054,0.0033,0.0045,0.0079)

input_data_as_np_array = np.asarray(input_data)

input_data_reshape = input_data_as_np_array.reshape(1,-1)

prediction = model.predict(input_data_reshape)

if prediction[0] == 'R':
    print("Rock")
else:
    print("Mine")
