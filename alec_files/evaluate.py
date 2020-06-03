import numpy as np
import pandas as pd

from pprint import pprint

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

import acquire as ac
import prepare as pr
import preprocessing as pp

def evaluate(model_object, X, y):
    """
    This function evaluates the classification model using the accuracy score given a model_object, X, and y
    """
    
    # calcuated accuracy
    accuracy = model_object.score(X, y)
    
    return accuracy

def append_evaluation(df, model_type, model_object, X, y, ):
    """
    This function does the following:
    1. Calls the evaluation function on the specified model_object, X, and y
    2. Creates a dictionary with the model type and accuracy score of the model
    3. Creates a new_evaluation DataFrame out of the dictionary created in the previous step
    4. Returns the evaluation DataFrame with the appended new_evaluation

    NOTE: This function assumes the existance of a DataFrame name `evaluation`
    """
    
    # calling evaluate function to calculate the RMSE of the model
    accuracy = evaluate(model_object, X, y)
    
    # creating dictionary with model specification and evaluation metric
    dictionary = {"model_type": [model_type], "accuracy": accuracy}
    
    # creating a DataFrame from the dictionary above
    new_evaluation = pd.DataFrame(dictionary)
    
    # appending values to the evaluation DataFrame and returning said DataFrame
    df = df.append(new_evaluation, ignore_index=True)

    return df