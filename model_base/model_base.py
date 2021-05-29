import os

import nltk
import numpy as np
import pandas as pd
from keras.models import load_model
from nltk.corpus import stopwords
from PIL import Image
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.python.ops.gen_array_ops import Empty
from wordcloud import WordCloud

from .clean_text import clean_text, delete_stopwords, lemmatize
from .sentiment_model import preprocessing_for_sent, sentiment_calculation
from .tf_func import final_topics

nltk.download(('stopwords'))


model = load_model('model_base/sentimen_model_full', compile=False)
CHANGE_QUERY_TEXT = "К сожалению, мы ничего не нашли по твоему запросу, попробуй изменить тему или тональность отзывов"

def get_df(file_name, user_id):
    columns = [
        'comment',
        'date_time',
        'color',
        'size',
        'thumb_up',
        'thumb_down',
        'prod_eval',
        'prod',
        'brand']
    if os.path.isfile(f'./df/{user_id}.csv'):
        os.remove(f'./df/{user_id}.csv')
    try:
        df = pd.read_json(file_name).transpose(
        ).reset_index().drop('index', axis=1)
    except BaseException:
        return CHANGE_QUERY_TEXT
    if df.empty or len(df) < 29:
        return CHANGE_QUERY_TEXT
    df = df.set_axis(columns, axis='columns')

    file_name = 'StopWords_extension.csv'
    path = f'./{file_name}'
    SW = pd.read_csv(path, index_col='Index')
    SW_list = SW['Word'].tolist()
    prod_text = clean_text(df['prod'][0])
    brand_text = clean_text(df['brand'][0])
    stopwords_add_by_category = [
        i for i in brand_text.split()] + [i for i in prod_text.split()]
    stopwords_add_by_category += lemmatize(stopwords_add_by_category)
    russian_stopwords = stopwords.words("russian")
    russian_stopwords.extend(stopwords_add_by_category + SW_list)

    df['cleaned_comment'] = df['comment'].map(lambda x: clean_text(x))
    df['cleaned_comment'] = df['cleaned_comment'].map(
        lambda x: delete_stopwords(x, russian_stopwords))
    df = df.drop(df[df['cleaned_comment'] == ''].index).reset_index(drop=True)
    df['lemma_comment'] = lemmatize(df['cleaned_comment'])
    df['lemma_comment'] = df['lemma_comment'].map(
        lambda x: delete_stopwords(x, russian_stopwords))
    df = df.drop(df[df['lemma_comment'] == ''].index)
    df = df.reset_index(drop=True)

    # tf_idf first
    vectorizer = TfidfVectorizer(min_df=2)
    vectorized_comments = vectorizer.fit_transform(df['lemma_comment'])
    #  creating a dictionary mapping the tokens to their tfidf values
    tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
        dict(tfidf), orient='index')
    tfidf.columns = ['tfidf']
    tfidf = tfidf.sort_values(by=['tfidf'], ascending=True)

    # Delete Stopwords. TFIDF once again
    max_lim = 6
    min_lim = 3
    SW_tfidf = tfidf[(tfidf['tfidf'] > max_lim) | (
        tfidf['tfidf'] < min_lim)].reset_index()['index'].tolist()
    russian_stopwords.extend(SW_tfidf)
    df['lemma_comment_2'] = df['lemma_comment'].map(
        lambda x: delete_stopwords(x, russian_stopwords))
    df = df.drop(df[df['lemma_comment_2'] == ''].index)
    df = df.reset_index(drop=True)
    df['words_count'] = df['lemma_comment_2'].apply(lambda x: len(x.split()))
    df = df.drop(df[df['words_count'] <= 3].index).reset_index(drop=True)
    if df.empty or len(df) < 29:
        return CHANGE_QUERY_TEXT
    df.to_csv(f'./df/{user_id}.csv', index=False)


def word_cloud_output(tf_idf_indexes, output_name):
    mask = np.array(Image.open('./berry.jpg'))
    unique_string = (" ").join(tf_idf_indexes)
    wordcloud = WordCloud(background_color='white',
                          mask=mask,
                          width=mask.shape[1],
                          height=mask.shape[0],
                          contour_width=3,
                          contour_color='red').generate(unique_string)
    wordcloud.to_file(output_name)


def tf_idf(file_name, user_id):

    config = {'SEED': 10,
              'clustering': {'affinity': 'rbf',
                             'count_max_clusters': 10,
                             'silhouette_metric': 'euclidean'},
              'comments': {'YOUTUBE_API_KEY': 'AIzaSyCPYNxHdsk6_-UX60p9Hm65cPXWXifut9A',
                           'count_video': 5,
                           'limit': 30,
                           'maxResults': 20,
                           'nextPageToken': '',
                           'query': 'дата сайенс'},
              'cross_val': {'test_size': 0.3},
              'dir_folder': '/Users/miracl6/airflow-mlflow-tutorial',
              'model': {'class_weight': 'balanced'},
              'model_lr': 'LogisticRegression',
              'model_vec': 'vector_tfidf',
              'name_experiment': 'my_first',
              'stopwords': 'russian',
              'tf_model': {'max_features': 1000}}

    try:
        df = pd.read_csv(f'./df/{user_id}.csv')
        n_clusters = 5

        def vectorize_text(data, tfidf):
            """
            Получение матрицы кол-ва слов в комменариях
            Очистка от пустых строк
            """
            # Векторизация
            X_matrix = tfidf.transform(data).toarray()
            return X_matrix

        comments_clean = df['lemma_comment_2']
        tfidf_clust = TfidfVectorizer(**config['tf_model']).fit(comments_clean)
        X_matrix = vectorize_text(comments_clean, tfidf_clust)

        clf = KMeans(
            n_clusters=n_clusters,
            max_iter=1000,
            n_init=1,
            random_state=10)
        clf.fit(X_matrix)
        cluster_labels = clf.labels_
        clust_centers = clf.cluster_centers_
        df_clust = pd.DataFrame(df['comment'].to_numpy(), columns=['comment'])
        df_clust['cluster'] = cluster_labels
        df_clust['lemma_comment_2'] = df['lemma_comment_2']
        df_clust = df_clust.reset_index()
        df_clust['vector'] = df_clust['index'].apply(lambda x: X_matrix[x])
        df_clust['sum_vec'] = df_clust['vector'].apply(lambda x: sum(x))
        df_clust = df_clust.drop(
            df_clust[df_clust['sum_vec'] == 0].index).reset_index(drop=True)
        df_clust = df_clust.drop(columns='index').reset_index()
        df_clust['centr'] = df_clust['cluster'].apply(
            lambda x: clust_centers[x])
        df_clust['dist'] = df_clust['index'].apply(
            lambda x: distance.euclidean(
                df_clust['centr'][x], df_clust['vector'][x]))

        def clust_topics_define(df):
            clust_num = df['cluster'].nunique()
            clust_topics = {}
            topl = []
            for i in range(clust_num):
                vectorizer = TfidfVectorizer(min_df=2)
                df_filt = df[df['cluster'] == i]['lemma_comment_2']
                vectorizer.fit_transform(df_filt)
                tfidf = dict(
                    zip(vectorizer.get_feature_names(), vectorizer.idf_))
                tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
                    dict(tfidf), orient='index')
                tfidf.columns = ['tfidf']
                tfidf = pd.DataFrame(
                    tfidf.sort_values(
                        by=['tfidf'],
                        ascending=True),
                    columns=['tfidf'])
                topl = final_topics(tfidf)
                for j in topl:
                    if j not in clust_topics.values():
                        clust_topics[i] = j
                        break
            return list(clust_topics.values())

        res = clust_topics_define(df_clust)
        preprocessed_comments = df['lemma_comment_2']
        vectorizer = TfidfVectorizer(min_df=2)
        vectorized_comments = vectorizer.fit_transform(preprocessed_comments)
        # creating a dictionary mapping the tokens to their tfidf values
        tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
        tfidf = pd.DataFrame(columns=['tfidf']).from_dict(
            dict(tfidf), orient='index')
        tfidf.columns = ['tfidf']
        tfidf = tfidf.sort_values(by=['tfidf'], ascending=True)
        if os.path.isfile(f'./topics/{user_id}.csv'):
            os.remove(f'./topics/{user_id}.csv')
        if os.path.isfile(f'./df_clust/{user_id}.csv'):
            os.remove(f'./df_clust/{user_id}.csv')
        pd.DataFrame(res).to_csv(f'./topics/{user_id}.csv', index=False)
        pd.DataFrame(df_clust).to_csv(f'./df_clust/{user_id}.csv', index=False)
        return res, tfidf.head(30).index
    except BaseException:
        return 'Мало комментариев'


# Функция для сохранения всех комментариев по определенным топикам(нужная
# для кнопки Показать еще)
def similar_comments_(word, nlp, user_id):
    df = pd.read_csv(f'./df/{user_id}.csv')
    if os.path.isfile(f'./topics/{user_id}.csv') and (
            word in list(pd.read_csv(f'./topics/{user_id}.csv')['0'])):
        num = list(pd.read_csv(f'./topics/{user_id}.csv')['0']).index(word)
        df_clust = pd.read_csv(f'./df_clust/{user_id}.csv')
        res = list(df_clust[df_clust['cluster'] == num].sort_values(
            by=['dist'], ascending=True)['comment'])
        res = pd.DataFrame(res, columns=['comment'])
        sen_pred = model.predict(preprocessing_for_sent(res))
        res = sentiment_calculation(res, preprocessing_for_sent(res), model)
        res['sentiment'] = res['comment']
    else:
        critical_similarity_value = 0.47        
        word_for_checking = nlp(word)  
        similarities = []
        for i in range(len(df['lemma_comment_2'])):
            similarities.append(
                nlp(df['lemma_comment_2'][i]).similarity(word_for_checking))
        if len(similarities) == 0:
            return
        df_temp = df.copy()
        df_temp[f'similarity_to_{word_for_checking}'] = similarities
        # сортировка по убыванию similarities, фильтрация в соответствии с
        # critical_similarity_value
        df_temp = df_temp.sort_values(
            by=f'similarity_to_{word_for_checking}',
            ascending=False)

        result = pd.DataFrame(df_temp[df_temp[f'similarity_to_{word_for_checking}'] > critical_similarity_value][[
                           'comment', f'similarity_to_{word_for_checking}']])      

        if preprocessing_for_sent(result) is 'No valid comments':            
            return

        res = sentiment_calculation(result, preprocessing_for_sent(result), model)        

    if res and len(res) > 0:
        if os.path.isfile(f'./similarity/{user_id}_{word}.csv'):
            os.remove(f'./similarity/{user_id}_{word}.csv')

        res.to_csv(f'./similarity/{user_id}_{word}.csv', index=False)
    else:
        return


def similar_comments(word, user_id):
    word_s = word[:-1]
    sent = word[-1]

    if os.path.isfile(f'./similarity/{user_id}_{word_s}.csv'):
        similar = pd.read_csv(f'./similarity/{user_id}_{word_s}.csv')
        os.remove(f'./similarity/{user_id}_{word_s}.csv')

        if sent == 'p':
            sentiment = 'Positive'
        if sent == 'n':
            sentiment = 'Negative'
        if sent != 'a':
            similar = similar[similar['Predicted_sent'] == sentiment]

        if len(similar) == 0:

            return CHANGE_QUERY_TEXT

        else:
            res = list(similar['comment'])[:10]

        if len(similar) > 10:
            similar = similar.loc[10:]
            similar.to_csv(f'./similarity/{user_id}_{word_s}.csv', index=False)

        if len(res) > 0:
            return '\n\n'.join(res)
        else:
            return CHANGE_QUERY_TEXT

    else:
        return CHANGE_QUERY_TEXT
