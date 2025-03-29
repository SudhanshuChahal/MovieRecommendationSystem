from surprise import Dataset, Reader,SVD
from collections import defaultdict


class CollaborativeFiltering:
    def __init__(self):
        self.model = SVD()
        self.trainset =None

    def train(self, trainset):
        self.trainset =trainset
        self.model.fit(self.trainset)

    def get_top_recommendations(self, user_id, k=10):
        if not self.trainset.knows_user(user_id):
            raise ValueError(f"User {user_id} not found")
        
        user_inner_id=self.trainset.to_inner_uid(user_id)
        user_ratings=set([iid for (iid, _) in self.trainset.ur[user_inner_id]])
        candidate_items = [iid for iid in self.trainset.all_items() if iid not in user_ratings]

        predictions =[]
        for iid in candidate_items[:1000]: 
            raw_iid =self.trainset.to_raw_iid(iid)
            pred =self.model.predict(user_id, raw_iid)
            predictions.append(pred)

        return sorted(predictions, key=lambda x: x.est, reverse=True)[:k]