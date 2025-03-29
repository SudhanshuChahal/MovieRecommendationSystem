import numpy as np 
from sklearn.metrics import mean_squared_error
from collections import defaultdict


def calculate_rmse(predictions):
    #"for collaborative filtering"
    actual = np.array([pred.r_ui for pred in predictions])
    predicted = np.array([pred.est for pred in predictions])
    return np.sqrt(mean_squared_error(actual, predicted))

def precision_at_k(recommendations,test_interactions, k=10):
    #"precision@k for recommendations"
    precisions =[]
    for user in recommendations:
        pred_items = set(recommendations[user][:k])
        true_items = set(test_interactions.get(user,[]))
        if len(pred_items)>0:
            prec = len(pred_items & true_items)/len(pred_items)
            precisions.append(prec)
    return np.mean(precisions) if precisions else 0.0

def recall_at_k(recommendations, test_interactions, k=10):
    #"recall@k for recommendations"
    recalls = []
    for user in recommendations:
        pred_items = set(recommendations[user][:k])
        true_items = set(test_interactions.get(user,[]))
        if len(true_items)> 0:
            rec = len(pred_items & true_items)/len(true_items)
            recalls.append(rec)
    return np.mean(recalls) if recalls else 0.0 

def get_test_interactions(testset):
    #"convert testset (list of predictions) to user-item interactions"
    test_interactions = defaultdict(list)
    for pred in testset:
        if pred.r_ui is not None:
            test_interactions[pred.uid].append(pred.iid)
    return test_interactions