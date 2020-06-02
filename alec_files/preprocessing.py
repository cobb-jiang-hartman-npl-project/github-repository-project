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

def split_repo_data(df, test_size=.2, train_size=.8, validate_size=.75, random_state=56):
    """
    This function takes in a DataFrame and returns a train and test DataFrame for exploration and modeling.
    I would like to stratify by language, but we need to get more than one observation per language in order to do so.
    This will work for now.

    TODO: Stratify or bin to remove stratification necessity
    """
    
    # call train_test_split on df
    train, test = train_test_split(df, test_size=test_size, train_size=train_size, random_state=random_state)

    # call train_test_split on the train
    train, validate = train_test_split(train, train_size=validate_size, random_state=random_state)

    return train, validate, test
