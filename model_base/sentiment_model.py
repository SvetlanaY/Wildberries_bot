import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
nltk.download('wordnet')
nltk.download('punkt')

def preprocessing_for_sent(df):
    ''' Takes dataframe after TFIDF2 and makes Series with comments prepared for model'''
    wn_lemmatizer = WordNetLemmatizer()
    lemmatized_text_for_sen = []
    for comment in df['comment']:
     #   print('in prep',comment.split())
        print(wn_lemmatizer.lemmatize('венчик'))
        lemmatized_text_for_sen.append(' '.join([wn_lemmatizer.lemmatize(word) for word in comment.split()]))

    if lemmatized_text_for_sen==[]:
        return 1
    print('lem',lemmatized_text_for_sen)
    for i in range(len(lemmatized_text_for_sen)):
        lemmatized_text_for_sen[i] = word_tokenize(lemmatized_text_for_sen[i])
    clean_tokenized_comment = [] 
    for i, element in enumerate(lemmatized_text_for_sen):     
        clean_tokenized_comment.append(' '.join([word for word in element]))    
    return pd.Series(clean_tokenized_comment)


def sentiment_calculation(df, series, model):
    '''
    returning origianl df with following collumns added:
    comment_for_sen - preprocessed comment
    Original_sent - original sentiment calculated as Positive in case of prod_eval >= 4 and Negative in other case
    pos_prob = probability of positive sentiment according to model
    Predicted_sent - predicted sentiment as Positive in case of pos_prob >= 0.6, Negative in case of pos_prob <= 0.6 and Neutral in othe case
    is_equal_sent - check whether Origianl sentiment is equal to predicted
    '''
    
    sen_pred = model.predict(series)   
    df['Predicted_sent'] = np.where(sen_pred >= 0.6, 'Positive', np.where(sen_pred <= 0.4, 'Negative', 'Neutral'))
    return df



    





    



