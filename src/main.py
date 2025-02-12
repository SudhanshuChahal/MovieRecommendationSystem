from data.data_loader import load_rating_data, load_movie_metadata
from data.preprocessor import clean_ratings, clean_movies, merge_data
import os
import pickle
from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBasedFiltering


rating_path = os.path.join("..", "data", "raw","ratings.csv" )
movies_path = os.path.join("..", "data", "raw","movies.csv" )

def main():
    
    # Load data
    ratings = load_rating_data(rating_path)
    movies = load_movie_metadata(movies_path)
    
    #preprocess
    ratings_clean =clean_ratings(ratings)
    movies_clean, tfidf_matrix = clean_movies(movies)
    merged_df = merge_data(ratings_clean, movies_clean)

    #save processed data
    ratings_clean.to_csv('../data/processed/ratings_clean.csv', index=False)
    movies_clean.to_csv('../data/processed/movies_clean.csv', index=False)
    merged_df.to_csv('../data/processed/merged_df.csv', index=False)

    #save TF-IDF matrix
    with open('../data/processed/tfidf_matrix.pkl','wb') as f:
        pickle.dump(tfidf_matrix, f)

    # Collaborative Filtering
    cf_model = CollaborativeFiltering()
    cf_model.train(ratings_clean)
    user_recommendations = cf_model.get_top_recommendations(user_id=1,k=5)
    print("collaborative Recommendations:", user_recommendations)

    #content based filetering
    with open('../data/processed/tfidf_matrix.pkl','rb') as f:
        tfidf_matrix = pickle.load(f)

    cb_model = ContentBasedFiltering(tfidf_matrix, movies_clean)
    content_recommendations = cb_model.get_recommendations(title='Toy Story', k=5)
    print("Content-Based Recommendations:",content_recommendations)



if __name__ == '__main__':
    main()