import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os

import spacy
from pymystem3 import Mystem
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import time

from model_base.clean_text import clean_text, lemmatize

from model_base.word_cloud import plot_wordcloud


def get_df(file_name,user_id):
    columns = ['comment', 'date_time', 'color','size', 'thumb_up', 'thumb_down', 'prod_eval', 'prod', 'brand']
    
    if os.path.isfile(f'./df/{user_id}.csv'):
        os.remove(f'./df/{user_id}.csv')

    try:
        df = pd.read_json(file_name).transpose().reset_index().drop('index', axis=1)
    except:
        return
    if df.empty or len(df) < 29 :
        return

    df = df.set_axis(columns, axis = 'columns')   

    # нужно обновить стоп-слова, добавив как миниму то, что в облаке. Сейчас использую стоп-слова NLTK, 
    # но стоит сравнить с другими
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(['очень', 'хороший', 'отличный', 'свой', 'отзыв', 'миксер', 'супер','это', 'спасибо', 'работа',
                re.sub(r'[.,?!@#~`$%^&*_+-=]', '', df['brand'][0].lower()), 
                re.sub(r'[.,?!@#~`$%^&*_+-=]', '', df['prod'][0].lower())])
    df['cleaned_comment'] = df['comment'].map(lambda x: clean_text(x,russian_stopwords))
    df[df['cleaned_comment']=='']
    df = df.drop(df[df['cleaned_comment']==''].index)
    df['lemma_comment'] = lemmatize(df['cleaned_comment'])
    df['lemma_comment'] = df['lemma_comment'].map(lambda x: clean_text(x,russian_stopwords))
    df = df.drop(df[df['lemma_comment']==''].index)
    df = df.reset_index(drop = True)
    
    
    df.to_csv(f'./df/{user_id}.csv',index=False)

      

def word_cloud_output(file_name,output_name,user_id):
    
    df = pd.read_csv(f'./df/{user_id}.csv')
    preprocessed_comments = df['lemma_comment']

    plot_wordcloud(preprocessed_comments, title="Word Cloud of comments")
    plt.savefig(output_name)
    return preprocessed_comments
    

def tf_idf(preprocessed_comments):
   try:
      vectorizer = TfidfVectorizer(min_df=10, ngram_range=(1, 2))
      vectorized_comments = vectorizer.fit_transform(preprocessed_comments)
      
   #  creating a dictionary mapping the tokens to their tfidf values
      tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
      tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
      tfidf.columns = ['tfidf']
      res = pd.DataFrame(tfidf.sort_values(by=['tfidf'], ascending=True).head(5))
      res = ', '.join(res.index)
      return res
   except:
      return 'Мало комментариев'   

def similar_comments(word,nlp,user_id):
    df = pd.read_csv(f'./df/{user_id}.csv')
    
    critical_similarity_value = 0.47    
    word_for_checking = nlp(word)
    similarities = []
    for i in range(len(df['lemma_comment'])):
        similarities.append(nlp(df['lemma_comment'][i]).similarity(word_for_checking))
    
    df_temp = df.copy()
    
    df_temp[f'similarity_to_{word_for_checking}'] = similarities
    #сортировка по убыванию similarities, фильтрация в соответствии с critical_similarity_value
    df_temp = df_temp.sort_values(by = f'similarity_to_{word_for_checking}', ascending = False).head(10)
    res = df_temp[df_temp[f'similarity_to_{word_for_checking}'] > critical_similarity_value][['comment', f'similarity_to_{word_for_checking}']]
    res = list(res['comment'])
    
    if len(res)>0:
        return '\n'.join(res)
    else: 
        return "По вашему запросу совпадений не найдено"
