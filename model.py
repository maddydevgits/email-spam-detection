import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def spam_email_prediction(input_mail):
    raw_mail_data = pd.read_csv('mail_data.csv')
    mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)),'')

    mail_data.loc[mail_data['Category'] == 'spam', 'Category',] = 0
    mail_data.loc[mail_data['Category'] == 'ham', 'Category',] = 1

    X = mail_data['Message']

    Y = mail_data['Category']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # transform the text data to feature vectors that can be used as input to the Logistic regression

    feature_extraction = TfidfVectorizer(min_df = 1, stop_words='english', lowercase='True')

    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # convert Y_train and Y_test values as integers

    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    model = LogisticRegression()
    # training the Logistic Regression model with the training data
    model.fit(X_train_features, Y_train)


    # convert text to feature vectors
    input_data_features = feature_extraction.transform(input_mail)

    # making prediction

    prediction = model.predict(input_data_features)
    if (prediction[0]==1):
        return 0
    else:
        return 1