import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import unicodedata

import re

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import acquire as ac

additional_stopwords = ["img", "1", "yes", "see", "width20", "height20", "okay_icon", "unknown"]

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
    string = re.sub(r"[^a-z0-9\s']", "", string)
    
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
    This function accepts a list of strings (lemmas) and returns a list and string after removing stopwords.
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

def quantify_lemmas(lemmas_sans_stopwords):
    """
    This function does the following:
    1. Takes in the list of lemmas_sans_stopwords returned from the remove_stopwords function
    2. Quantifies the length of each readme in the form of the list of lemmas
    3. Returns a list of integers representing the length of the lemmas
    """

    # quantify lemmas
    len_of_clean_readme = len(lemmas_sans_stopwords)

    return len_of_clean_readme

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

    # quantify lemmas
    dictionary["len_of_clean_readme_contents"] = len(lemmas_sans_stopwords)
    
    return dictionary

def wrangle_readme_data(extra_stopwords=additional_stopwords, exclude_stopwords=[]):
    """
    This function does the following:
    1. Reads the data.json file into a pandas DataFrame
    2. Converts the DataFrame to a list of dictionaries
    3. Iterates over the list_of_dictionaries object using a list comprehension calling the `prep_readme` function on each dictionary
    4. Converts the list of dictionaries produced by the list comprehension to a pandas DataFrame
    5. Masks DataFrame to only include observations where the language is not null
    6. Masks DataFrame to exclude lower outliers
    7. Use lambda function to exclude languages with only one observation
    8. Resets DataFrame index
    9. Drops original index column
    10. Returns the resultant DataFrame
    
    Parameters:
    extra_stopwords: Extra words can be added the standard english stopwords using the extra_stopwords parameter.
    exclude_stopwords: Words can be excluded from the standard english stopwords using the exclude_stopwords parameter.

    TODO: Write resultant DataFrame to disk
    """
    
    # read data.json into a DataFrame
    df = pd.read_json("data.json")

    # convert DataFrame to a list of dictionaries
    list_of_dictionaries = df.to_dict("records")

    # list comprehension applying prep_article function to each dictionary
    list_of_dictionaries = [prep_readme(dictionary, key="readme_contents", extra_stopwords=extra_stopwords, exclude_stopwords=exclude_stopwords) for dictionary in list_of_dictionaries]

    # convert list_of_dictionaries to DataFrame
    df = pd.DataFrame(list_of_dictionaries)

    # mask DataFrame to only include observations where the language is not null
    df = df[df.language.isna() == False]

    # remove outliers using mask
    df = df[df.len_of_clean_readme_contents >= 10]

    # use lambda function to exclude languages with only one observation in order to properly stratify
    df = df.groupby('language').filter(lambda x : len(x) >= 2)

    # reset DataFrame index
    df.reset_index(inplace=True)

    # drop original index column
    df.drop(columns=["index"], inplace=True)

    return df