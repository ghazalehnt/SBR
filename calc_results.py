import argparse
import csv
import json
import time
from collections import Counter, defaultdict

import transformers
import pandas as pd

from SBR.utils.metrics import calculate_ranking_metrics_macro_avg_over_qid

relevance_level = 1

goodreads_rating_mapping = {
    None: None,  ## this means there was no rating
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def get_metrics(ground_truth, prediction_scores, ranking_metrics=None, calc_pytrec=True):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics_macro_avg_over_qid(gt=ground_truth, pd=prediction_scores,
                                                           relevance_level=relevance_level,
                                                           given_ranking_metrics=ranking_metrics,
                                                           calc_pytrec=calc_pytrec)
    print(f"ranking metrics in {time.time() - start}")
    return results


def filter_users_by_len(train_ds, min_user_review_len=None, review_field=None):
    train_user_count = Counter(train_ds['user_id'])
    # keep only users with "long" reviews if given len
    if min_user_review_len is not None:
        train_ds[review_field] = train_ds[review_field].fillna("")  # TODO multiple fields?
        keep_users = {}
        tokenizer = transformers.AutoTokenizer.from_pretrained(BERTMODEL)
        user_reviews = defaultdict(list)
        for user_id, review in zip(train_ds['user_id'], train_ds[review_field]):
            user_reviews[user_id].append(review)
        for user_id in user_reviews:
            user_reviews[user_id] = ". ".join(user_reviews[user_id])
            num_toks = len(tokenizer(user_reviews[user_id], truncation=False)['input_ids'])
            if num_toks >= min_user_review_len:
                keep_users[user_id] = num_toks
        train_user_count = {k: v for k, v in train_user_count.items() if k in keep_users.keys()}
    return train_user_count


def group_users_threshold(train_user_count, thresholds):
    groups = {thr: set() for thr in sorted(thresholds)}
    if len(thresholds) > 0:
        groups['rest'] = set()
        for user in train_user_count:
            added = False
            for thr in sorted(thresholds):
                if train_user_count[user] <= thr:
                    groups[thr].add(str(user))
                    added = True
                    break
            if not added:
                groups['rest'].add(str(user))

    ret_group = {}
    last = 1
    for gr in groups:
        if gr == 'rest':
            new_gr = f"{last}+"
        else:
            new_gr = f"{last}-{gr}"
            last = gr + 1
        ret_group[new_gr] = groups[gr]
    return ret_group


def group_users_ratios(train_user_count, ratios):
    sorted_users = [k for k, v in sorted(train_user_count.items(), key=lambda x: x[1])]  # sporadic to bibliophilic
    n_users = len(sorted_users)
    if len(ratios) != 3:
        raise ValueError("3 ratios must be given")
    cnts = [int(r*n_users) for r in ratios]
    if sum(cnts) < n_users:
        return ValueError("check here1")
    elif sum(cnts) > n_users:
        return ValueError("check here2")
    groups = {}
    groups["sporadic"] = set(sorted_users[:cnts[0]])
    groups["regular"] = set(sorted_users[cnts[0]:-cnts[2]])
    groups["bibliophilic"] = set(sorted_users[-cnts[2]:])
    return groups


def get_results(prediction, ground_truth, ranking_metrics, thresholds, ratios,
                train_file=None, min_user_review_len=None, review_field=None):
    ret = []
    start = time.time()
    # we may have some users who only exist in training set (not in all datasets)
    train_ds = pd.read_csv(train_file, dtype=str)
    filtered_train_user_count = filter_users_by_len(train_ds, min_user_review_len, review_field)
    user_groups = {}
    if thresholds is not None:
        user_groups = group_users_threshold(filtered_train_user_count, thresholds)
    # filtered_train_user_count = {str(k): v for k, v in filtered_train_user_count.items()} ?? why did we have this?
    if ratios is not None:
        user_groups = group_users_ratios(filtered_train_user_count, ratios)

    if len(filtered_train_user_count) == 0:
        return
    print(f"grouped users in {time.time()-start}")

    # apply filters to the gt and pd
    ground_truth = {u: v for u, v in ground_truth.items() if u in filtered_train_user_count}
    prediction = {u: v for u, v in prediction.items() if u in filtered_train_user_count}

    # only count the number of users and interactions in this given eval file (for example when changes such as rev len ...)
    start = time.time()
    # users:
    num_filtered_eval_users = len(ground_truth.keys())
    num_filtered_eval_users_group = {group: 0 for group in user_groups}
    for g in user_groups:
        num_filtered_eval_users_group[g] = len(set(ground_truth.keys()).intersection(user_groups[g]))
    ret.append({"#total-users": num_filtered_eval_users})
    ret.extend([{f"#users-{g}":  num_filtered_eval_users_group[g]} for g in user_groups])
    # interactions:
    pos_total_cnt = 0
    neg_total_cnt = 0
    pos_inter_cnt = {group: 0 for group in user_groups}
    neg_inter_cnt = {group: 0 for group in user_groups}
    for u in ground_truth:
        pos_total_cnt += len([k for k, v in ground_truth[u].items() if v == 1])
        neg_total_cnt += len([k for k, v in ground_truth[u].items() if v == 0])

        group = None
        for g in user_groups:
            if u in user_groups[g]:
                group = g
                break
        if group is not None:
            pos_inter_cnt[group] += len([k for k, v in ground_truth[u].items() if v == 1])
            neg_inter_cnt[group] += len([k for k, v in ground_truth[u].items() if v == 0])
    ret.append({"#total-pos-inter": pos_total_cnt})
    ret.append({"#total-neg-inter": neg_total_cnt})
    ret.extend([{f"#pos-inter-{g}": pos_inter_cnt[g]} for g in user_groups])
    ret.extend([{f"#neg-inter-{g}": neg_inter_cnt[g]} for g in user_groups])
    print(f"count inters {time.time() - start}")

    # calc metrics:
    # total:
    total_results = get_metrics(ground_truth=ground_truth,
                                prediction_scores=prediction,
                                ranking_metrics=ranking_metrics)
    metric_header = sorted(total_results.keys())
    ret.append({"ALL":  {h: total_results[h] for h in metric_header}})

    # groups:
    for g in user_groups:
        group_results = get_metrics(ground_truth={k: v for k, v in ground_truth.items() if k in user_groups[g]},
                                    prediction_scores={k: v for k, v in prediction.items() if k in user_groups[g]},
                                    ranking_metrics=ranking_metrics)
        metric_header = sorted(group_results.keys())
        ret.append({g:  {h: group_results[h] for h in metric_header}})

    return ret


if __name__ == "__main__":
    # hard coded
    BERTMODEL = "bert-base-uncased"  # TODO hard coded

    parser = argparse.ArgumentParser()

    # required: path to gt and pd to be evaluated:
    parser.add_argument('--pos_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--neg_gt_path', type=str, default=None, help='path to gt file')
    parser.add_argument('--pred_path', type=str, default=None, help='path to pd file')
    parser.add_argument('--out_path', type=str, default=None, help='path to output file')
    parser.add_argument('--train_file_path', type=str, default=None, help='path to similar runs config file')

    # optional: threshold to groupd users:
    parser.add_argument('--thresholds', type=int, nargs='+', default=None, help='user thresholds')

    # optional: ratio of user groups:
    parser.add_argument('--ratios', type=int, nargs='+', default=None, help='user thresholds')

    # optional if we want to only calculate the metrics for users with certain review length.
    parser.add_argument('--min_user_review_len', type=int, default=None, help='min length of the user review')
    parser.add_argument('--review_field', type=str, default=None, help='review field')
    args, _ = parser.parse_known_args()

    ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "P_1", "P_5", "recip_rank"]

    thrs = args.thresholds
    rates = args.ratios
    if thrs is not None and rates is not None:
        raise ValueError("either ratios or thresholds not both")
    if rates is not None:
        if sum(rates) != 100:
            raise ValueError(f"rations must sum up to 100: {rates}")
        rates = [r/100 for r in rates]

    prediction = json.load(open(args.pred_path))
    if len(prediction.keys()) == 1 and "predicted" in prediction:
        prediction = prediction["predicted"]
    pos_file = pd.read_csv(args.pos_gt_path, dtype=str)
    neg_file = pd.read_csv(args.neg_gt_path, dtype=str)
    ground_truth = defaultdict(lambda: defaultdict())
    for user_id, item_id in zip(pos_file["user_id"], pos_file["item_id"]):
        ground_truth[user_id][item_id] = 1  # TODO if we are doing rating or something else change this
    for user_id, item_id in zip(neg_file["user_id"], neg_file["item_id"]):
        ground_truth[user_id][item_id] = 0  # TODO if we are doing rating or something else change this

    results = get_results(prediction, ground_truth, ranking_metrics, thrs, rates,
                          train_file=args.train_file_path,
                          min_user_review_len=args.min_user_review_len,
                          review_field=args.review_field)
    results.append({"min_user_review_len": args.min_user_review_len, "review_field": args.review_field})

    outfile = args.out_path
    print(outfile)
    outfile_f = open(outfile, "w")
    for line in results:
        json.dump(line, outfile_f)
        outfile_f.write("\n\n")
    outfile_f.close()