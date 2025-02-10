import pandas as pd

def load_rating_data(data_path):
    ratings_df = pd.read_csv(data_path)
    return ratings_df

def load_movie_metadata(data_path):
    movies_df = pd.read_csv(data_path)
    return movies_df