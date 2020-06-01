import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import unicodedata

import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import og_acquire as og_ac
import acquire as ac

def basic_clean(string: str) -> str:
    """
    This function accepts a string and returns the string after applying some basic text cleaning to each word.
    """
    
    # lowercase all characters
    string = string.lower()
    
    # normalize unicode characters
    string = unicodedata.normalize("NFKD", string)\
                .encode("ascii", "ignore")\
                .decode("utf-8", "ignore")
    
    # replace anything that is not a letter, number, whitespace or a single quote.
    string = re.sub(r"[^a-z0-9\s]", "", string)
    
    return string

def tokenize(string: str) -> list:
    """
    This function accepts a string and returns a list of tokens after tokenizing to each word.
    """
    
    # make tokenizer object
    tokenizer = ToktokTokenizer()

    # use tokenizer object and return string
    list_of_tokens = tokenizer.tokenize(string, return_str=False)
    
    return list_of_tokens

def lemmatize(list_of_tokens: list) -> list:
    """
    This function accepts a list of tokens and returns a list after applying lemmatization to each word.
    """
    
    # create lemmatizer object
    wnl = nltk.stem.WordNetLemmatizer()
    
    # use lemmatizer to generate list of stems
    lemmas = [wnl.lemmatize(word) for word in list_of_tokens]
    
    # join lemmas to whitespace to create a cohesive string
    cohesive_lemmas = " ".join(lemmas)
    
    return lemmas

def remove_stopwords(lemmas, extra_stopwords=[], exclude_stopwords=[]):
    """
    This function accepts a list of strings (lemmas) and returns a list after removing stopwords.
    Extra words can be added the standard english stopwords using the extra_stopwords parameter.
    Words can be excluded from the standard english stopwords using the exclude_stopwords parameter.
    """
    
    # create stopword list
    stopword_list = stopwords.words("english")
    
    # extend extra_stopwords variable to stopwords if there are words in the parameter
    if not extra_stopwords:
        stopword_list
    else:
        stopword_list.extend(extra_stopwords)
    
    # remove words in exclude_stopwords variable from stopwords if there are words in the parameter
    if not exclude_stopwords:
        stopword_list
    else:
        stopword_list = [word for word in stopword_list if word not in exclude_stopwords]
    
    # list comprehension 
    lemmas_sans_stopwords = [word for word in lemmas if word not in stopword_list]

    # join lemmas_or_stems_sans_stopwords to whitespace to return a cohesive string
    string_sans_stopwords = " ".join(lemmas_sans_stopwords)

    return lemmas_sans_stopwords, string_sans_stopwords

def prep_readme(dictionary, key, extra_stopwords=[], exclude_stopwords=[]):
    """
    This function accepts a dictionary representing a singular repository containing a readme, as specified 
    in the key parameter, to clean. 
    The function then returns a dictionary containing the cleaned text.
    """
    
    # indexing the original readme_contents
    readme_contents = dictionary[key]
    
    # running basic_clean function on readme_contents
    cleaned_readme_content = basic_clean(readme_contents)
    
    # running tokenize function on cleaned_readme_contents
    readme_tokens = tokenize(cleaned_readme_content)
    
    # running lemmatize function on readme_tokens
    readme_lemmas = lemmatize(readme_tokens)
    
    # running remove_stopwords on readme_lemmas
    lemmas_sans_stopwords, string_sans_stopwords = remove_stopwords(readme_lemmas, extra_stopwords=extra_stopwords, exclude_stopwords=exclude_stopwords)
    
    # creating cleaned column in dictionary
    dictionary["clean_readme_contents"] = string_sans_stopwords
    
    return dictionary

def wrangle_readme_data(extra_stopwords=[], exclude_stopwords=[]):
    """
    This function does the following:
    1. Reads the data.json file into a pandas DataFrame
    2. Converts the DataFrame to a list of dictionaries
    3. Iterates over the list_of_dictionaries object using a list comprehension calling the `prep_readme` function on each dictionary
    4. Converts the list of dictionaries produced by the list comprehension to a pandas DataFrame
    5. Returns the resultant DataFrame
    
    Parameters:
    extra_stopwords: Extra words can be added the standard english stopwords using the extra_stopwords parameter.
    exclude_stopwords: Words can be excluded from the standard english stopwords using the exclude_stopwords parameter.
    """
    
    # read data.json into a DataFrame
    df = pd.read_json("data.json")

    # convert DataFrame to a list of dictionaries
    list_of_dictionaries = df.to_dict("records")

    # list comprehension applying prep_article function to each dictionary
    list_of_dictionaries = [prep_readme(dictionary, key="readme_contents", extra_stopwords=extra_stopwords, exclude_stopwords=exclude_stopwords) for dictionary in list_of_dictionaries]

    # convert list_of_dictionaries to DataFrame
    df = pd.DataFrame(list_of_dictionaries)

    return df