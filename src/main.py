from data.data_loader import load_rating_data, load_movie_metadata
import os


rating_path = os.path.join("..", "data", "raw","ratings.csv" )
movies_path = os.path.join("..", "data", "raw","movies.csv" )

def main():
    
    # Load data
    ratings = load_rating_data(rating_path)
    movies = load_movie_metadata(movies_path)
    print("ratings data\n",ratings.head(),"\nmovies metadata\n",movies.head())


if __name__ == '__main__':
    main()