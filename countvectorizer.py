# Hotel Reviews Sentiment Analysis Project
# This Project Uses Count Vectorizer for the models
# @author Simran
# @version 1.0

import pandas as pd  # analyse data
import numpy as np  # for working with arrays
# machine learning library
from sklearn.model_selection import train_test_split  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# >>- importing dataset ->>
print('>>- importing dataset ->>')
df = pd.read_csv('Datafiniti_Hotel_Reviews.csv')

# >>- getting rid of null values ->>
print('>>- getting rid of null values ->>')
df = df.dropna()

# >>- taking 30% representative sample ->>
print('>>- taking 30% representative sample ->>')
np.random.seed(34)
df1 = df.sample(frac=0.3)

# >>- adding sentiments column ->>
print('>>- adding sentiments column ->>')
df1['sentiments'] = df1.rating.apply(lambda x: 0 if x in [1, 2] else 1)

# >>- defining input training features and labels ->>
print('>>- defining input training features and labels ->>')
X = df1['reviews']  # input feature
Y = df1['sentiments']  # label

# >>- Using Count Vectorizer ->>
print('>>- Using Count Vectorizer ->>')
# use count vectorizer to vectorize the text data in
# ->review column and use three different
# ->classification models from scikit-learn models.

# -> split dataset into training sets and testing sets <-
X_train, X_test, Y_train, Y_test = \
    train_test_split(X, Y, test_size=0.5, random_state=24)

# Vectorizing the text data
print('Vectorizing the text data')
cv = CountVectorizer()
ctmTr = cv.fit_transform(X_train)
X_test_dtm = cv.transform(X_test)

# >- Logistic Regression ->
print('>- Logistic Regression ->')

# Training the model
print('Training the model')
lr = LogisticRegression()
lr.fit(ctmTr, Y_train)

# Generating Accuracy score
print('Generating Accuracy score')
lr_score = lr.score(X_test_dtm, Y_test)
print("Results for Logistic Regression with CountVectorizer")
print(lr_score)

# Predicting the labels for the test data
print('Predicting the labels for the test data')
Y_pred_lr = lr.predict(X_test_dtm)

# Setting up Confusion matrix
print('Setting up Confusion matrix')
cm_lr = confusion_matrix(Y_test, Y_pred_lr)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_lr).ravel()
print(tn, fp, fn, tp)

# Printing True Positive and Negative rates
print('Printing True Positive and Negative rates')
tpr_lr = round(tp / (tp + fn), 4)
tnr_lr = round(tn / (tn + fp), 4)
print(tpr_lr, tnr_lr)

# >- Support Vector Machine ->
print('>- Support Vector Machine ->')

# Training the model
print('Training the model')
svcl = svm.SVC()
svcl.fit(ctmTr, Y_train)

# Generating Accuracy score
print('Generating Accuracy score')
svcl_score = svcl.score(X_test_dtm, Y_test)
print('Results for Support Vector Machine with CountVectorizer')
print(svcl_score)

# Predicting the labels for the test data
print('Predicting the labels for the test data')
Y_pred_sv = svcl.predict(X_test_dtm)

# Setting up Confusion matrix
print('Setting up Confusion matrix')
cm_sv = confusion_matrix(Y_test, Y_pred_sv)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_sv).ravel()
print(tn, fp, fn, tp)

# Printing True Positive and Negative rates
print('Printing True Positive and Negative rates')
tpr_sv = round(tp/(tp + fn), 4)
tnr_sv = round(tn/(tn + fp), 4)
print(tpr_sv, tnr_sv)

# >- K Nearest Neighbor ->
print('>- K Nearest Neighbor ->')

# Training the model
print('Training the model')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(ctmTr, Y_train)

# Generating Accuracy score
print('Generating Accuracy score')
knn_score = knn.score(X_test_dtm, Y_test)
print('Results for K Nearest Neighbor with CountVectorizer')
print(knn_score)

# Predicting the labels for the test data
print('Predicting the labels for the test data')
Y_pred_knn = knn.predict(X_test_dtm)

# Setting up Confusion matrix
print('Setting up Confusion matrix')
cm_knn = confusion_matrix(Y_test, Y_pred_knn)
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_knn).ravel()
print(tn, fp, fn, tp)

# Printing True Positive and Negative rates
print('Printing True Positive and Negative rates')
tpr_knn = round(tp/(tp + fn), 4)
tnr_knn = round(tn/(tn + fp), 4)
print(tpr_knn, tnr_knn)
