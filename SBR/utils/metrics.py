# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)

import pytrec_eval
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

ranking_metrics = [
    "P_1",
    "P_5",
    "P_10",
    "P_20",
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "recall_5",
    "recall_10",
    "recall_20",
]


def calculate_metrics(ground_truth, prediction_scores, users, items, relevance_level=1, prediction_threshold=0.5):
    result = calculate_ranking_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, items, relevance_level)

    temp = calculate_cl_metrics_micro(ground_truth, prediction_scores, prediction_threshold)
    result.update(temp)

    temp = calculate_cl_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, prediction_threshold)
    result.update(temp)

    return result


def calculate_ranking_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, items, relevance_level):
    # qid= user1:{ item1:1 } ...
    gt = {str(u): {} for u in set(users)}
    pd = {str(u): {} for u in set(users)}
    for i in range(len(ground_truth)):
        gt[str(users[i])][str(items[i])] = int(ground_truth[i])
        pd[str(users[i])][str(items[i])] = float(prediction_scores[i])
    evaluator = pytrec_eval.RelevanceEvaluator(gt, ranking_metrics, relevance_level=int(relevance_level))
    scores = [[metrics_dict.get(m, -1) for m in ranking_metrics] for metrics_dict in evaluator.evaluate(pd).values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(ranking_metrics, scores))
    return scores


def get_p_r_f1(ground_truth, predictions):
    return [precision_score(ground_truth, predictions), \
           recall_score(ground_truth, predictions), \
           f1_score(ground_truth, predictions)]


def calculate_cl_metrics_micro(ground_truth, prediction_scores, prediction_threshold):
    predictions = [1 if p > prediction_threshold else 0 for p in prediction_scores]
    scores = get_p_r_f1(ground_truth, predictions)
    return {"precision_micro": scores[0], "recall_micro": scores[1], "f1_micro": scores[2]}


def calculate_cl_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, prediction_threshold):
    gt_user = {u: [] for u in set(users)}
    pd_user = {u: [] for u in set(users)}
    for i in range(len(ground_truth)):
        gt_user[users[i]].append(ground_truth[i])
        pd_user[users[i]].append(1 if prediction_scores[i] > prediction_threshold else 0)

    scores = {u: get_p_r_f1(gt_user[u], pd_user[u]) for u in set(users)}
    scores = np.array(list(scores.values())).mean(axis=0).tolist()
    return {"precision_macro": scores[0], "recall_macro": scores[1], "f1_macro": scores[2]}
