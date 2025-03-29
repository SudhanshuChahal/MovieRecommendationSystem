from data.data_loader import load_rating_data, load_movie_metadata
from data.preprocessor import clean_ratings, clean_movies, merge_data
import os
import pickle
from models.collaborative import CollaborativeFiltering
from models.content_based import ContentBasedFiltering
from evaluation.metrics import calculate_rmse, precision_at_k, recall_at_k, get_test_interactions 
from surprise.model_selection import train_test_split 
from surprise import Dataset, Reader


rating_path = os.path.join("..", "data", "raw","ratings.csv" )
movies_path = os.path.join("..", "data", "raw","movies.csv" )

def main():
    #Create directories if missing
    os.makedirs('../data/processed',exist_ok=True)

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

    # split testset and trainset
    reader =Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_clean[['userId','movieId','rating']],reader)
    trainset, testset = train_test_split(data, test_size=0.25) 

    # Collaborative Filtering
    cf_model = CollaborativeFiltering()
    cf_model.train(trainset)

    # Generate recommendations for user 1(demo)
    try:
        user_recommendations = cf_model.get_top_recommendations(user_id=1, k=5)
        print("\nCollaborative Recommendations for user 1: ", user_recommendations)
    except ValueError as e:
        print(f"\nError for User 1:{e}")

    #Evaluate on Test Set (real ratings)
    predictions = cf_model.model.test(testset)
    rmse = calculate_rmse(predictions)
    print(f"\nCollaborative Filtering RMSE: {rmse:.3f}")

    #Generate recommendations for all TEST users
    test_users = set([pred.uid for pred in predictions])
    all_user_recommendations = {}
    for user in test_users:
        try:
            user_recs = cf_model.get_top_recommendations(user, k=10)
            all_user_recommendations[user] = [pred.iid for pred in user_recs]
        except ValueError:
            continue

    # Calculate metrics using REAL TEST INTERACTIONS
    test_interactions = get_test_interactions(predictions)
    prec = precision_at_k(all_user_recommendations, test_interactions, k=10)
    rec = recall_at_k(all_user_recommendations, test_interactions, k=10)
    print(f"Collaborative filtering Precision@10: {prec:.3f}")
    print(f"Collaborative Filetering Recall@10:{rec:.3f}")

    #content based filetering
    with open('../data/processed/tfidf_matrix.pkl','rb') as f:
        tfidf_matrix = pickle.load(f)

    cb_model = ContentBasedFiltering(tfidf_matrix, movies_clean)

    seed_movie='Toy Story'
    liked_movies = ['Antz','Monsters, Inc.']

    try:
        content_recs = cb_model.get_recommendations(title='Toy Story', k=10)
        print(f"\nContent-Based Recommendations for '{seed_movie}':{content_recs}")

        content_prec= len(set(content_recs) & set(liked_movies))/ len(content_recs)
        print(f"Content-Based Precisions@10 (vs liked movies): {content_prec:.3f}")
    except ValueError as e:
        print(f"\nContent-Based Error:{e}")

if __name__ == '__main__':
    main()