%matplotlib inline
%load_ext autoreload
%autoreload 2
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
from pprint import pprint

# scraping modules
from requests import get
from bs4 import BeautifulSoup

import unicodedata
import re
import json

from wordcloud import WordCloud

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings("ignore")

import acquire
import og_acquire
import prepare


def get_others(df):
    '''
    input pandas series with value counts, convert it into a dataframe 
    to get new dataframe with dataframe with others category which is the 
    sum of the data past the first 13 rows
    '''
    df.reset_index()
    df2 = df[12:].sum()
    df2 = pd.DataFrame(df2).reset_index()
    df2 = df2.rename(columns={0:'language'})
    df = df.append(df2)
    df = df.drop(columns=(['index']))
    df = df.rename(index={0:'other'})
    df = df.sort_values(by='language', ascending=False)
    data = df.iloc[:13]
    return data


def clean(text: str) -> list:
    'A simple function to cleanup text data'
    wnl = nltk.stem.WordNetLemmatizer()
    stopwords = nltk.corpus.stopwords.words('english')
    text = (text.encode('ascii', 'ignore')
             .decode('utf-8', 'ignore')
             .lower())
    words = re.sub(r'[^\w\s]', '', text).split() # tokenization
    return [wnl.lemmatize(word) for word in words if word not in stopwords]



def word_freq(df):
    '''
    creates a list of words by language
    drops them into a series & turns them into a dataframe showing
    the most frequent words per language
    '''
    # lang_words
    go_words = ' '.join(df2[df2.language == 'Go'].clean_readme_contents).split()
    python_words = ' '.join(df2[df2.language == 'Python'].clean_readme_contents).split()
    javascript_words = ' '.join(df2[df2.language == 'JavaScript'].clean_readme_contents).split()
    java_words = ' '.join(df2[df2.language == 'Java'].clean_readme_contents).split()
    c_words = ' '.join(df2[df2.language == 'C'].clean_readme_contents).split()
    cplus_words = ' '.join(df2[df2.language == 'C++'].clean_readme_contents).split()
    HTML_words = ' '.join(df2[df2.language == 'HTML'].clean_readme_contents).split()
    Jupyter_words = ' '.join(df2[df2.language == 'Jupyter Notebook'].clean_readme_contents).split()
    Vue_words = ' '.join(df2[df2.language == 'Vue'].clean_readme_contents).split()
    CSS_words = ' '.join(df2[df2.language == 'CSS'].clean_readme_contents).split()
    shell_words = ' '.join(df2[df2.language == 'Shell'].clean_readme_contents).split()
    obj_words = ' '.join(df2[df2.language == 'Objective C'].clean_readme_contents).split()
    rust_words = ' '.join(df2[df2.language == 'Rust'].clean_readme_contents).split()
    Csharp_words = ' '.join(df2[df2.language == 'C#'].clean_readme_contents).split()
    kotlin_words = ' '.join(df2[df2.language == 'Kotlin'].clean_readme_contents).split()
    r_words = ' '.join(df2[df2.language == 'R'].clean_readme_contents).split()
    dart_words = ' '.join(df2[df2.language == 'Dart'].clean_readme_contents).split()
    scala_words = ' '.join(df2[df2.language == 'Scala'].clean_readme_contents).split()
    powershell_words = ' '.join(df2[df2.language == 'Powershell'].clean_readme_contents).split()
    Rascal_words = ' '.join(df2[df2.language == 'Rascal'].clean_readme_contents).split()
    tex_words = ' '.join(df2[df2.language == 'TeX'].clean_readme_contents).split()
    groovy_words = ' '.join(df2[df2.language == 'Groovy'].clean_readme_contents).split()
    apache_words = ' '.join(df2[df2.language == 'ApacheConf'].clean_readme_contents).split()
    PHP_words = ' '.join(df2[df2.language == 'PHP'].clean_readme_contents).split()
    type_words = ' '.join(df2[df2.language == 'TypeScript'].clean_readme_contents).split()
    all_words = ' '.join(df2.clean_readme_contents).split
    
    # Series
    go_freq = pd.Series(go_words).value_counts()
    python_freq = pd.Series(python_words).value_counts()
    javascript_freq = pd.Series(javascript_words).value_counts()
    java_freq = pd.Series(java_words).value_counts()
    c_freq = pd.Series(c_words).value_counts()
    cplus_freq = pd.Series(cplus_words).value_counts()
    html_freq = pd.Series(HTML_words).value_counts()
    jup_freq = pd.Series(Jupyter_words).value_counts()
    vue_freq = pd.Series(Vue_words).value_counts()
    css_freq = pd.Series(CSS_words).value_counts()
    shell_freq = pd.Series(shell_words).value_counts()
    obj_freq = pd.Series(obj_words).value_counts()
    rust_freq = pd.Series(rust_words).value_counts()
    csharp_freq = pd.Series(Csharp_words).value_counts()
    kotlin_freq = pd.Series(kotlin_words).value_counts()
    r_freq = pd.Series(r_words).value_counts()
    dart_freq = pd.Series(dart_words).value_counts()
    scala_freq = pd.Series(scala_words).value_counts()
    powershell_freq = pd.Series(powershell_words).value_counts()
    rascal_freq = pd.Series(Rascal_words).value_counts()
    tex_freq = pd.Series(tex_words).value_counts()
    groovy_freq = pd.Series(groovy_words).value_counts()
    apache_freq = pd.Series(apache_words).value_counts()
    php_freq = pd.Series(PHP_words).value_counts()
    type_freq = pd.Series(type_words).value_counts()
    all_freq = pd.Series(all_words).value_counts()
    
    # Concatting into df
    word_counts = (pd.concat([all_freq, go_freq, python_freq, javascript_freq, java_freq,
                              c_freq, cplus_freq, html_freq, jup_freq, vue_freq, css_freq,
                              shell_freq, obj_freq, rust_freq, csharp_freq, kotlin_freq,
                              r_freq, dart_freq, scala_freq, powershell_freq, rascal_freq,
                              tex_freq, groovy_freq, apache_freq, php_freq, type_freq 
                             ], axis=1, sort=True)
              .set_axis(['all', 'go', 'python', 'javascript', 'java', 'c', 'c++', 'html', 'jupyter notebook',
                        'vue', 'css', 'shell', 'objective c', 'rust', 'c#', 'kotlin', 'r', 'dart', 'scala', 'powershell',
                        'rascal', 'tex', 'groovy', 'apacheconf', 'php', 'typescript'], axis=1, inplace=False)
              .fillna(0)
              .apply(lambda s: s.astype(int)))
    
    return word_counts
    