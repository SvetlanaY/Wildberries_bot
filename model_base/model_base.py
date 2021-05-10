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

from model_base.clean_text import clean_text, lemmatize, delete_stopwords

from model_base.word_cloud import plot_wordcloud
from model_base.tf_func import pos_define, final_topics


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

    file_name = 'StopWords_extension.csv'
    path = f'./{file_name}'
    SW = pd.read_csv(path, index_col = 'Index')
    SW_list = SW['Word'].tolist()
    prod_text = clean_text(df['prod'][0])
    brand_text = clean_text(df['brand'][0])
    stopwords_add_by_category = [i for i in brand_text.split()]+[i for i in prod_text.split()]

    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(stopwords_add_by_category+SW_list)

    df['cleaned_comment'] = df['comment'].map(lambda x: clean_text(x))
    df['cleaned_comment'] = df['cleaned_comment'].map(lambda x: delete_stopwords(x,russian_stopwords))
    df = df.drop(df[df['cleaned_comment']==''].index).reset_index(drop = True)

    df['lemma_comment'] = lemmatize(df['cleaned_comment'])
    df['lemma_comment'] = df['lemma_comment'].map(lambda x: delete_stopwords(x,russian_stopwords))
    df = df.drop(df[df['lemma_comment']==''].index)
   
    df = df.reset_index(drop = True)

    #tf_idf first
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    vectorized_comments = vectorizer.fit_transform(df['lemma_comment'])
    #  creating a dictionary mapping the tokens to their tfidf values
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']
   # tfidf = tfidf.sort_values(by=['tfidf'], ascending=True)
    tfidf['pos'] = tfidf.index.map(lambda x: pos_define(x))
    #tfidf1_res = pd.DataFrame(final_topics(tfidf), columns = ['tfidf'])

    #Delete Stopwords. TFIDF once again
    max_lim = 6
    min_lim = 3
    SW_tfidf = tfidf[(tfidf['tfidf']>max_lim)|(tfidf['tfidf']<min_lim)].reset_index()['index'].tolist()
    russian_stopwords.extend(SW_tfidf)
    df['lemma_comment_2'] = df['lemma_comment'].map(lambda x: delete_stopwords(x,russian_stopwords))
    df = df.drop(df[df['lemma_comment_2']==''].index)
    df = df.reset_index(drop = True)
   # preprocessed_comments_2 = df['lemma_comment_2']


    
    
    df.to_csv(f'./df/{user_id}.csv',index=False)

      

def word_cloud_output(file_name,output_name,user_id):
    
    df = pd.read_csv(f'./df/{user_id}.csv')
    preprocessed_comments = df['lemma_comment_2']

    plot_wordcloud(preprocessed_comments)
    plt.savefig(output_name)
    return preprocessed_comments
    

def tf_idf(preprocessed_comments):
   
   try:
       
      vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
       
      vectorized_comments = vectorizer.fit_transform(preprocessed_comments)
      
   #  creating a dictionary mapping the tokens to their tfidf values
      tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
      
      tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
               
      tfidf.columns = ['tfidf']
      tfidf['pos'] = tfidf.index.map(lambda x: pos_define(x))
      
      #Эти слова показываем пользователю, он вводит то, по чему хочет почитать подробнее, или свое слово
      res = pd.DataFrame(tfidf[(tfidf['pos']!='V') & (tfidf['pos'] !='ADV')].sort_values(by=['tfidf'], ascending=True).head(5))
      print(pd.DataFrame(tfidf[(tfidf['pos']!='V') & (tfidf['pos'] !='ADV')].sort_values(by=['tfidf'], ascending=True)))
      #res = pd.DataFrame(tfidf.sort_values(by=['tfidf'], ascending=True).head(5))
      
      
     # res = ', '.join(res.index)
      return res.index
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
        return '\n\n'.join(res)
       
    else: 
        return "По вашему запросу совпадений не найдено"
