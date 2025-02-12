from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


class ContentBasedFiltering:
    def __init__(self, tfidf_matrix, movies_df):
        self.tfidf_matrix = tfidf_matrix
        self.movies_df = movies_df
        self.cosine_sim =cosine_similarity(tfidf_matrix, tfidf_matrix)

    def get_recommendations(self, title, k=10):
        idx = self.movies_df[self.movies_df['title'] == title].index[0]
        sim_scores =list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
        sim_indices = [i[0] for i in sim_scores[1:k+1]]

        return self.movies_df['title'].iloc[sim_indices].tolist()