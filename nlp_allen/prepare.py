import unicodedata
import re
import json
import pandas as pd

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

def basic_clean(string:str):
    #Lowercase everything
    string = string.lower()
    #Normalize unicode characters
    string = unicodedata.normalize('NFKD', string)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')
    #Replace anything that is not a letter, number, whitespace or a single quote.
    string = re.sub(r"[^a-z0-9\s']", '', string)
    return string

def tokenize(clean_string:str):
    tokenizer = nltk.tokenize.ToktokTokenizer()
    string = tokenizer.tokenize(clean_string, return_str=True)
    return string

def lemmatize(tokenized_string:str):
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in tokenized_string.split()]
    return lemmas

def remove_stopwords(lemmatized_string):
    stopword_list = stopwords.words('english')
    stopword_removed_string =  [word for word in lemmatized_string if word not in stopword_list]
    return ' '.join(stopword_removed_string) 

def prepare_text(original:str):
    clean_text = basic_clean(original)
    tokenized_text= tokenize(clean_text)
    lemmatized_text=lemmatize(tokenized_text)
    prepared_text = remove_stopwords(lemmatized_text)
    return prepared_text

def prepare_data():
    df = pd.read_json('data.json')
    df['readme_contents']= df.readme_contents.apply(prepare_text)
    return df.dropna()