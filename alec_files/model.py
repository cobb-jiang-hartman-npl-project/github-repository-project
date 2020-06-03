import numpy as np
import pandas as pd

import unicodedata

import re

import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import acquire as ac
import prepare as pr
import preprocessing as pp

def create_vectorizer_features_and_target():
    """
    This function does the following:
    1. Wrangles README data into a DataFrame
    2. Creates a tfidf vectorizer object
    3. Fits and transforms the clean_readme_contents Series from the df to create features
    4. Establishes a model target
    5. Splits the features (X) and target (y) into train and test
    6. Returns the vectorizer, train features, and train target to be used to fit the model
    """
    

    # wrangle data to train model
    df = pr.wrangle_readme_data()

    # create tfidf vectorizer object
    tfidf = TfidfVectorizer()

    # use tfidf object to create model features
    X = tfidf.fit_transform(df.clean_readme_contents)

    # establish model target
    y = df.language

    # split data
    X_train, X_test, y_train, y_test = pp.split_repo_data(X, y)

    return tfidf, X_train, y_train

def create_and_fit_model(features, target):
    """
    This function does the following:
    1. Creates model object
    2. Fits model on features and target
    3. Returns model object
    """    

    # create model object
    tree = DecisionTreeClassifier(max_depth=5, random_state=56)

    # fit model object
    tree.fit(features, target)
    
    # return model object to be used later to predict
    return tree

def predict_language(string: str) -> str:
    """
    This function does the following:
    1. Calls create_vectorizer_features_and_target function to get data to fit model
    and vectorizer to later transform input of string argument.
    2. Calls create_and_fit_model to get model that will predict language of string input
    3. Prepares the input of the string argument, the text of a README file, for modeling by calling the following
    functions:
        a. pr.basic_clean
        b. pr.tokenize
        c. pr.lemmatize
        d. pr.remove_stopwords + additional_stopwords
    4. Creates features (X) out of the string_sans_stopwords variable using the tfidf vectorizer object
    to transform
    5. Predicts the language of the features (X)
    6. Index numpy.ndarray object for string of predicted_language
    7. Returns the language_as_string variable
    """
    
    # call create_vectorizer_features_and_target function to get data to fit model
    tfidf, features, target = create_vectorizer_features_and_target()

    # call fit_model to get model that will predict language of string input
    model = create_and_fit_model(features, target)
    
    # call pr.basic_clean
    string = pr.basic_clean(string)

    # call pr.tokenize
    list_of_tokens = pr.tokenize(string)

    # call pr.lemmatize
    list_of_lemmas = pr.lemmatize(list_of_tokens)

    # additional_stopwords variable
    additional_stopwords = ["img", "1", "yes", "see", "width20", "height20", "okay_icon", "unknown", "http", "org", "com", "www", "md", "doc", "github", "svg"]
    
    # call pr.remove_stopwords
    lemmas_sans_stopwords, string_sans_stopwords = pr.remove_stopwords(list_of_lemmas, extra_stopwords=additional_stopwords, exclude_stopwords=[])

    # create X variable for model as vectorized string_sans_stopwords
    X = tfidf.transform([string_sans_stopwords])
    
    # predict language of README
    predicted_language = model.predict(X)    
    
    # index numpy.ndarray object for string of language
    language_as_string = predicted_language[0]

    return language_as_string