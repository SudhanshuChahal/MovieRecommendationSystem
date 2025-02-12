from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_ratings(ratings_df):
    ratings_clean = ratings_df.drop(columns=['timestamp'])

    return ratings_clean
 

def clean_movies(movies_df):
    movies_clean = movies_df.copy()

    movies_clean['genres'] =movies_clean['genres'].str.split('|')

    movies_clean['year'] = movies_clean['title'].str.extract(r'\((\d{4})\)')
    movies_clean['year'] = pd.to_numeric(movies_clean['year'], errors='coerce')

    movies_clean['title'] = movies_clean['title'].str.replace(r'\s*\(\d{4}\)','',regex=True)

    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(movies_clean['genres']),columns=mlb.classes_)
    movies_clean = pd.concat([movies_clean, genres_encoded],axis=1)

    tfidf =TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_clean['genres'].apply(lambda x: ' '.join(x)))

    return movies_clean,tfidf_matrix


def merge_data(ratings_df, movies_df):
    merged_df = pd.merge(ratings_df, movies_df, on='movieId', how='left')

    return merged_df
