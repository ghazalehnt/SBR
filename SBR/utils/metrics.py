# how to do when micro averaging in cross validation ask andrew (do micro for each fold and then avg? or "concat" all results and do micro for all?)
import json
from collections import defaultdict

import pytrec_eval
from sklearn.metrics import ndcg_score
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

ranking_metrics = [
    "ndcg_cut_5",
    "ndcg_cut_10",
    "ndcg_cut_20",
    "P_1",
    "recip_rank"
]


def calculate_metrics(ground_truth, prediction_scores, users, items, relevance_level, given_ranking_metrics=None):
    # # qid= user1:{ item1:1 } ...
    gt = {str(u): {} for u in set(users)}
    pd = {str(u): {} for u in set(users)}
    min_not_zero = 1
    for i in range(len(ground_truth)):
        if len(items) == 0:
            gt[str(users[i])][str(i)] = float(ground_truth[i])
            pd[str(users[i])][str(i)] = float(prediction_scores[i])
        else:
            gt[str(users[i])][str(items[i])] = float(ground_truth[i])
            pd[str(users[i])][str(items[i])] = float(prediction_scores[i])
        if ground_truth[i] != 0 and ground_truth[i] < min_not_zero:
            min_not_zero = ground_truth[i]
    return calculate_ranking_metrics_macro_avg_over_users(gt, pd, relevance_level, given_ranking_metrics,
                                                          True if min_not_zero!=1 else False)


def calculate_ranking_metrics_macro_avg_over_users(gt, pd, relevance_level,
                                                   given_ranking_metrics=None, weighted_label=False):
    if given_ranking_metrics is None:
        given_ranking_metrics = ranking_metrics
    if weighted_label:
        # weighted ground truth
        results = calculate_ndcg(gt, pd, [m for m in given_ranking_metrics if m.startswith("ndcg_")])
    else:
        gt = {k: {k2: int(v2) for k2, v2 in v.items()} for k, v in gt.items()}
        results = calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics)
    return results


def calculate_ranking_metrics_pytreceval(gt, pd, relevance_level, given_ranking_metrics):
    '''
    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    evaluator = pytrec_eval.RelevanceEvaluator(gt, given_ranking_metrics, relevance_level=int(relevance_level))
    per_user_scores = evaluator.evaluate(pd)
    scores = [[metrics_dict.get(m, -1) for m in given_ranking_metrics] for metrics_dict in per_user_scores.values()]
    scores = np.array(scores).mean(axis=0).tolist()
    scores = dict(zip(given_ranking_metrics, scores))
    return scores


def ndcg(gt, pd, k):
    per_user_socre = []
    for user in gt.keys():
        true_rel = np.asarray([[v for k, v in gt[user].items()]])
        pred = np.asarray([[v for k, v in pd[user].items()]])
        per_user_socre.append(ndcg_score(true_rel, pred, k=k))
    return per_user_socre


def calculate_ndcg(gt, pd, given_ranking_metrics):
    '''

    :param gt: dict of user -> item -> true score (relevance)
    :param pd: dict of user -> item -> predicted score
    :param relevance_level:
    :param given_ranking_metrics:
    :return: metric scores
    '''
    ret = defaultdict()
    for m in given_ranking_metrics:
        if m.startswith("ndcg_cut_"):
            per_user_scores = ndcg(gt, pd, int(m[m.rindex("_")+1:]))
        else:
            raise NotImplementedError("other metrics not implemented")
        ret[m] = np.array(per_user_scores).mean().tolist()
    return ret


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
        gt[str(user_ids[i])][str(item_ids[i])] = float(ground_truth[i])
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
