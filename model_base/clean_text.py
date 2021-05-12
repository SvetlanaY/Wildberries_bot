import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

import spacy
from pymystem3 import Mystem
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import time


def delete_non_letters(words):
    new_words = []
    words = words.split()
    
    for word in words:
        new_word = "".join(c if c.isalpha() else " " for c in word )
        
        if new_word != '':
            new_words.append(new_word)
    text = ' '.join(c for c in new_words)
        
    return text


### Text Normalizing function. Part of the following function was taken from this link. 
def clean_text(text):
   
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

    #удаляет пунктуацию
    #translation_table = str.maketrans("", "", string.punctuation)
    #text = text.translate(translation_table)   
    
    text = text.lower()
    ## Clean the text
    text = re.sub(r"[,_»«\*!.\/'+-=)(]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\%", " ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"\|", " ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r'"', " ", text)
    text = re.sub(r'«', " ", text)
    text = re.sub(r'\*', " ", text)
    text = re.sub(r'\?', " ", text)
    text = re.sub(r'»', " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = emoji_pattern.sub(r'', text)

    text = text.split()
    text = ' '.join(re.sub("[^А-Яа-яё]",'', i) for i in text)
    
    
    #text = " ".join(text.split())
    
    text = delete_non_letters(text)
    
    return text


def delete_stopwords(text,russian_stopwords): 
    text = text.split()
    text = [w for w in text if not w in russian_stopwords and len(w) >= 3]
    text = " ".join(text)
    
    return text


def lemmatize(text): 
    text =  [i for i in text]
    m = Mystem()
    merged_text = "|".join(text)

    doc = []
    res = []
    count = 0
    lemma = m.lemmatize(merged_text)
    for t in lemma:
        
        if '|' not in t and count + 1 < len(lemma):
            doc.append(t)
            count += 1 
          
        else:
            doc = ''.join(i for i in doc)
            res.append(doc)
            count += 1
            doc = []
    return res

