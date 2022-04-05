# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)
import json

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
    return calculate_ranking_metrics(gt, pd, relevance_level)


def calculate_ranking_metrics(gt, pd, relevance_level):
    '''

    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :return: metric scores
    '''
    evaluator = pytrec_eval.RelevanceEvaluator(gt, ranking_metrics, relevance_level=int(relevance_level))
    scores = [[metrics_dict.get(m, -1) for m in ranking_metrics] for metrics_dict in evaluator.evaluate(pd).values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(ranking_metrics, scores))
    return scores


def get_p_r_f1(ground_truth, predictions):
    return [precision_score(ground_truth, predictions, zero_division=0), \
           recall_score(ground_truth, predictions), \
           f1_score(ground_truth, predictions)]


def calculate_cl_metrics_micro(ground_truth, prediction_scores, prediction_threshold):
    predictions = [1 if p > prediction_threshold else 0 for p in prediction_scores]
    return calculate_cl_micro(ground_truth, predictions)


def calculate_cl_micro(ground_truth, predictions):
    '''
    :param ground_truth: true classes (relevance)
    :param predictions: predicted relevance
    :return: micro p,r,f1
    '''
    scores = get_p_r_f1(ground_truth, predictions)
    return {"precision_micro": scores[0], "recall_micro": scores[1], "f1_micro": scores[2]}


def calculate_cl_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, prediction_threshold):
    gt_user = {u: [] for u in set(users)}
    pd_user = {u: [] for u in set(users)}
    for i in range(len(ground_truth)):
        gt_user[users[i]].append(ground_truth[i])
        pd_user[users[i]].append(1 if prediction_scores[i] > prediction_threshold else 0)
    return calculate_cl_macro(gt_user, pd_user)


def calculate_cl_macro(gt_user, pd_user):
    '''
    :param gt_user: per user true class (relevant or not)
    :param pd_user: per user predicted class
    :return: macro p,r,f1
    '''
    scores = {u: get_p_r_f1(gt_user[u], pd_user[u]) for u in gt_user.keys()}
    scores = np.array(list(scores.values())).mean(axis=0).tolist()
    return {"precision_macro": scores[0], "recall_macro": scores[1], "f1_macro": scores[2]}


def log_results(output_path, ground_truth, prediction_scores, internal_user_ids, internal_items_ids,
                external_users, external_items):
    # we want to log the results corresponding to external user and item ids
    ex_users = external_users.to_pandas().set_index("internal_user_id")
    user_ids = ex_users.loc[internal_user_ids].user_id.values
    ex_items = external_items.to_pandas().set_index("internal_item_id")
    item_ids = ex_items.loc[internal_items_ids].item_id.values

    gt = {str(u): {} for u in set(user_ids)}
    pd = {str(u): {} for u in set(user_ids)}
    for i in range(len(ground_truth)):
        gt[str(user_ids[i])][str(item_ids[i])] = int(ground_truth[i])
        pd[str(user_ids[i])][str(item_ids[i])] = float(prediction_scores[i])
    json.dump({"ground_truth": gt, "predicted": pd}, open(output_path, 'w'))
