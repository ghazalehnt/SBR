# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)
import json
from collections import defaultdict

# import pytrec_eval
from sklearn.metrics import ndcg_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

# ranking_metrics = [
#     "P_1",  # min number of pos interaction for users in test is 1, so P@higher than 1 may penalty even if it shouldn't
#     "P_5",
#     "P_10",
#     "P_20",
#     "ndcg_cut_5",
#     "ndcg_cut_10",
#     "ndcg_cut_20",
#     "recall_100",
#     "recall_1000",
#     "Rprec",
#     "recip_rank"
# ]

ranking_metrics = [
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
]

def calculate_metrics(ground_truth, prediction_scores, users, items,
                      relevance_level=1, prediction_threshold=0.5, ranking_only=False):
    # prediction_scores is sigmoid applied already.
    result = calculate_ranking_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, items, relevance_level)
    # if ranking_only:
    #     return result
    #
    # predictions = (prediction_scores > prediction_threshold).float()
    #
    # temp = calculate_cl_metrics_micro(ground_truth, predictions, prediction_threshold)
    # result.update(temp)
    #
    # temp = calculate_cl_metrics_macro_avg_over_users(ground_truth, predictions, users, prediction_threshold)
    # result.update(temp)

    return result

# todo later simplify these many functions, depending on which metrics to implement
def calculate_ranking_metrics_macro_avg_over_users(ground_truth, prediction_scores, users, items, relevance_level,
                                                   given_ranking_metrics=None):
    # # qid= user1:{ item1:1 } ...
    gt = {str(u): {} for u in set(users)}
    pd = {str(u): {} for u in set(users)}
    for i in range(len(ground_truth)):
        gt[str(users[i])][str(items[i])] = float(ground_truth[i])
        pd[str(users[i])][str(items[i])] = float(prediction_scores[i])
    return calculate_ranking_metrics(gt, pd, relevance_level, given_ranking_metrics)


# def calculate_ranking_metrics(gt, pd, relevance_level, given_ranking_metrics=None):
#     '''
#
#     :param gt: dict of user -> item -> true score (relevance)
#     :param pd: dict of user -> item -> predicted score
#     :param relevance_level:
#     :param given_ranking_metrics:
#     :return: metric scores
#     '''
#     if given_ranking_metrics is None:
#         given_ranking_metrics = ranking_metrics
#     evaluator = pytrec_eval.RelevanceEvaluator(gt, given_ranking_metrics, relevance_level=int(relevance_level))
#     per_user_scores = evaluator.evaluate(pd)
#     scores = [[metrics_dict.get(m, -1) for m in given_ranking_metrics] for metrics_dict in per_user_scores.values()]
#     scores = np.array(scores).mean(axis=0).tolist()
#     scores = dict(zip(given_ranking_metrics, scores))
#     return scores


def ndcg(gt, pd, k):
    per_user_socre = []
    for user in gt.keys():
        true_rel = np.asarray([[v for k, v in gt[user].items()]])
        pred = np.asarray([[v for k, v in pd[user].items()]])
        per_user_socre.append(ndcg_score(true_rel, pred, k=k))
    return per_user_socre


# def precision_k_macro_users_weighted(gt, pd, k):
#     max_gt = 1
#     for items in gt.values():
#         for scores in items.values():
#             if max(scores) > max_gt:
#                 max_gt = max(scores)
#     if max_gt > 1:
#         raise ValueError("This is not implemented for the ratings dataset, have to set a relevance-level ...")
#
#     scores = []
#     for user, items in gt.items():
#         gt_items = sorted(items.values(), reverse=True)
# todo contiue, it is a bit weird, as we have relevance threshold 0.5 also, and if gt=0.1 and pd=0.5 what's the deal ...


def calculate_ranking_metrics(gt, pd, relevance_level, given_ranking_metrics=None):
    '''

    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    ret = defaultdict()
    for m in given_ranking_metrics:
        if m.startswith("ndcg_cut_"):
            per_user_scores = ndcg(gt, pd, int(m[m.rindex("_")+1:]))
        # elif m.startswith("P_"):
        #     score = precision_k_macro_users_weighted(gt, pd, int(m[m.rindex("_") + 1:]))
        else:
            raise NotImplementedError("other metrics not implemented")
        ret[m] = np.array(per_user_scores).mean().tolist()
    return ret


# def get_p_r_f1(ground_truth, predictions, avg="binary"):
#     return [precision_score(ground_truth, predictions, zero_division=0, average=avg), \
#            recall_score(ground_truth, predictions, average=avg), \
#            f1_score(ground_truth, predictions, average=avg)]


# def calculate_cl_metrics_micro(ground_truth, predictions, prediction_threshold):
#     return calculate_cl_micro(ground_truth, predictions)


# def calculate_cl_micro(ground_truth, predictions):
#     '''
#     :param ground_truth: true classes (relevance)
#     :param predictions: predicted relevance
#     :return: micro p,r,f1
#     '''
#     scores = get_p_r_f1(ground_truth, predictions)
#     return {"precision_micro": scores[0], "recall_micro": scores[1], "f1_micro": scores[2]}


# def calculate_cl_metrics_macro_avg_over_users(ground_truth, predictions, users, prediction_threshold):
#     gt_user = {u: [] for u in set(users)}
#     pd_user = {u: [] for u in set(users)}
#     for i in range(len(ground_truth)):
#         gt_user[users[i]].append(int(ground_truth[i]))
#         pd_user[users[i]].append(int(predictions[i]))
#
#     return calculate_cl_macro(gt_user, pd_user)


# def calculate_cl_macro(gt_user, pd_user):
#     '''
#     :param gt_user: per user true class (relevant or not)
#     :param pd_user: per user predicted class
#     :return: macro p,r,f1
#     '''
#     scores = {u: get_p_r_f1(gt_user[u], pd_user[u]) for u in gt_user.keys()}
#     scores = np.array(list(scores.values())).mean(axis=0).tolist()
#     return {"precision_macro": scores[0], "recall_macro": scores[1], "f1_macro": scores[2]}


def log_results(output_path, ground_truth, prediction_scores, internal_user_ids, internal_items_ids,
                external_users, external_items):
    # we want to log the results corresponding to external user and item ids
    ex_users = external_users.to_pandas().set_index(INTERNAL_USER_ID_FIELD)
    user_ids = ex_users.loc[internal_user_ids].user_id.values
    ex_items = external_items.to_pandas().set_index(INTERNAL_ITEM_ID_FIELD)
    item_ids = ex_items.loc[internal_items_ids].item_id.values

    gt = {str(u): {} for u in sorted(set(user_ids))}
    pd = {str(u): {} for u in sorted(set(user_ids))}
    for i in range(len(ground_truth)):
        gt[str(user_ids[i])][str(item_ids[i])] = float(ground_truth[i])  #TODO float?
        pd[str(user_ids[i])][str(item_ids[i])] = float(prediction_scores[i])
    json.dump({"predicted": pd}, open(output_path['predicted'], 'w'))
    json.dump({"ground_truth": gt}, open(output_path['ground_truth'], 'w'))
    cnt = 0
    if 'log' in output_path and 'text' in ex_users.columns:
        with open(output_path["log"], "w") as f:
            for user_id in gt.keys():
                if cnt == 100:
                    break
                cnt += 1
                f.write(f"user:{user_id} - text:{ex_users[ex_users['user_id'] == user_id]['text'].values[0]}\n\n\n")
                for item_id, pd_score in sorted(pd[user_id].items(), key=lambda x:x[1], reverse=True):
                    f.write(f"item:{item_id} - label:{gt[user_id][item_id]} - score:{pd_score} - text:{ex_items[ex_items['item_id'] == item_id]['text'].values[0]}\n\n")
                f.write("-----------------------------\n")
