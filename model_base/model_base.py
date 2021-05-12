import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from PIL import Image

import spacy
from pymystem3 import Mystem
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

import string
import time

from .clean_text import clean_text, lemmatize, delete_stopwords

from .tf_func import pos_define, final_topics


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
    stopwords_add_by_category += lemmatize(stopwords_add_by_category)
    
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
    vectorizer = TfidfVectorizer(min_df=2)
    vectorized_comments = vectorizer.fit_transform(df['lemma_comment'])
    #  creating a dictionary mapping the tokens to their tfidf values
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']

    tfidf = tfidf.sort_values(by=['tfidf'], ascending=True)
   # tfidf['pos'] = tfidf.index.map(lambda x: pos_define(x))
    #tfidf1_res = pd.DataFrame(final_topics(tfidf), columns = ['tfidf'])

    #Delete Stopwords. TFIDF once again
    max_lim = 6
    min_lim = 3
    SW_tfidf = tfidf[(tfidf['tfidf']>max_lim)|(tfidf['tfidf']<min_lim)].reset_index()['index'].tolist()
    russian_stopwords.extend(SW_tfidf)
    df['lemma_comment_2'] = df['lemma_comment'].map(lambda x: delete_stopwords(x,russian_stopwords))
    df = df.drop(df[df['lemma_comment_2']==''].index)
    df = df.reset_index(drop = True)
    df['words_count'] = df['lemma_comment_2'].apply(lambda x: len(x.split()))
    df = df.drop(df[df['words_count']<=3].index).reset_index(drop = True)
    
       
    df.to_csv(f'./df/{user_id}.csv',index=False)

      

def word_cloud_output(tf_idf_indexes,output_name):
    mask = np.array(Image.open('./berry.jpg'))
    unique_string=(" ").join(tf_idf_indexes)
    wordcloud = WordCloud(background_color='white', 
                          mask=mask,
                          width = mask.shape[1], 
                          height = mask.shape[0],
                          contour_width=3,
                          contour_color='red').generate(unique_string)
      
    wordcloud.to_file(output_name)


def tf_idf(file_name,user_id):
   
   try: 
      df = pd.read_csv(f'./df/{user_id}.csv')
      preprocessed_comments = df['lemma_comment_2']
       
      vectorizer = TfidfVectorizer(min_df=2)
       
      vectorized_comments = vectorizer.fit_transform(preprocessed_comments)
      
   #  creating a dictionary mapping the tokens to their tfidf values
      tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
      
      tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
               
      tfidf.columns = ['tfidf']
      tfidf = tfidf.sort_values(by=['tfidf'], ascending=True)

      res = pd.DataFrame(final_topics(tfidf), columns = ['tfidf'])
     # tfidf['pos'] = tfidf.index.map(lambda x: pos_define(x))
      
      #Эти слова показываем пользователю, он вводит то, по чему хочет почитать подробнее, или свое слово
     # res = pd.DataFrame(tfidf[(tfidf['pos']!='V') & (tfidf['pos'] !='ADV')].sort_values(by=['tfidf'], ascending=True).head(5))
    #  print(pd.DataFrame(tfidf[(tfidf['pos']!='V') & (tfidf['pos'] !='ADV')].sort_values(by=['tfidf'], ascending=True)))
      #res = pd.DataFrame(tfidf.sort_values(by=['tfidf'], ascending=True).head(5))
      
      
     # res = ', '.join(res.index)
      print(res)
      return res['tfidf'],tfidf.head(30).index
   except:
      return 'Мало комментариев'   

def similar_comments_(word,nlp,user_id):
    
    df = pd.read_csv(f'./df/{user_id}.csv')
    
    critical_similarity_value = 0.47    
    word_for_checking = nlp(word)
    similarities = []
    for i in range(len(df['lemma_comment_2'])):
        similarities.append(nlp(df['lemma_comment_2'][i]).similarity(word_for_checking))
    
    df_temp = df.copy()
    
    df_temp[f'similarity_to_{word_for_checking}'] = similarities
    #сортировка по убыванию similarities, фильтрация в соответствии с critical_similarity_value
    df_temp = df_temp.sort_values(by = f'similarity_to_{word_for_checking}', ascending = False)
    #.loc[10*count:(count+1)*count]
    res =pd.DataFrame(df_temp[df_temp[f'similarity_to_{word_for_checking}'] > critical_similarity_value][['comment', f'similarity_to_{word_for_checking}']])
    if len(res)>0:
        res.to_csv(f'./similarity/{user_id}_{word}.csv',index=False)


def similar_comments(word,nlp,user_id):

    
    if os.path.isfile(f'./similarity/{user_id}_{word}.csv'):
        similar = pd.read_csv(f'./similarity/{user_id}_{word}.csv')
        os.remove(f'./similarity/{user_id}_{word}.csv')


        res = list(similar['comment'])[:10]
        if len(similar)>10:
            similar=similar.loc[10:]
            similar.to_csv(f'./similarity/{user_id}_{word}.csv',index=False)    
    
        if len(res) > 0:
            return '\n\n'.join(res)
        else: 
            return "По вашему запросу совпадений не найдено"    
       
    else: 
        return "По вашему запросу совпадений не найдено"
