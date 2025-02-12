from surprise import Dataset, Reader,SVD
from collections import defaultdict


class CollaborativeFiltering:
    def __init__(self):
        self.model = SVD(
            n_factors=100,
            n_epochs=100,
            lr_all=0.01,
            reg_all=0.1
        )

    def train(self, ratings_df):
        reader = Reader(rating_scale = (1,5))
        data = Dataset.load_from_df(
            ratings_df[['userId','movieId','rating']],

        reader
        )
        self.trainset = data.build_full_trainset()
        self.model.fit(self.trainset)

    def get_top_recommendations(self, user_id, k=10):
        if user_id not in self.trainset.all_users():
            raise ValueError(f"User {user_id} not found")
        testset = self.trainset.build_anti_testset()
        predictions  = self.model.test(testset)
        user_predictions = [pred for pred in predictions if pred.uid == user_id]

        return sorted(user_predictions, key=lambda x: x.est, reverse=True) [:k]