import numpy as np
import pandas as pd

import unicodedata

import re

import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split

import acquire as ac
import prepare as pr

def split_repo_data(*arrays, test_size=.2, train_size=.8, random_state=56):
    """
    This function takes in a DataFrame and returns a train and test DataFrame for exploration and modeling.
    I would like to stratify by language, but we need to get more than one observation per language in order to do so.
    This will work for now.

    TODO: Stratify or bin to remove stratification necessity
    TODO: Validate split
    """
    
    # call train_test_split on df
    X_train, y_train, X_test, y_test = train_test_split(*arrays, test_size=test_size, train_size=train_size, random_state=random_state)

    # need to join X and y train for a singular train DataFrame if you want a validate split

    # call train_test_split on the train
    # train, validate = train_test_split(train, train_size=validate_size, random_state=random_state)

    return X_train, y_train, X_test, y_test
