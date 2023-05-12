import argparse
import json
import time
from collections import Counter, defaultdict

import transformers
import pandas as pd
import numpy as np

from SBR.utils.metrics import calculate_ranking_metrics_detailed

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
    results = calculate_ranking_metrics_detailed(gt=ground_truth, pd=prediction_scores,
                                                 relevance_level=relevance_level,
                                                 given_ranking_metrics=ranking_metrics,
                                                 calc_pytrec=calc_pytrec)
    # micro avg:
    micro_res = defaultdict()
    for m in results:
        assert len(results[m]) == len(ground_truth)
        micro_res[m] = np.array(results[m]).mean().tolist()
    # macro avg per user:
    macro_res = defaultdict()
    for m in results:
        user_qids_res = defaultdict(list)
        for qid, r in zip(ground_truth.keys(), results[m]):
            uid = qid[:qid.index("_")]
            user_qids_res[uid].append(r)
        user_res = []
        for uid in user_qids_res:
            user_res.append(np.array(user_qids_res[uid]).mean().tolist())
        macro_res[m] = np.array(user_res).mean().tolist()
    print(f"ranking metrics in {time.time() - start}")
    return micro_res, macro_res


def filter_users_by_len(train_ds, min_user_review_len=None, review_field=None):
    train_user_count = Counter(train_ds['user_id'])
    # keep only users with "long" reviews if given len
    if min_user_review_len is not None:
        train_ds[review_field] = train_ds[review_field].fillna("")  # TODO multiple fields?
        keep_users = {}
        tokenizer = transformers.AutoTokenizer.from_pretrained(BERTMODEL)
        user_reviews = defaultdict(list)
        for uid, review in zip(train_ds['user_id'], train_ds[review_field]):
            user_reviews[uid].append(review)
        for uid in user_reviews:
            user_reviews[uid] = ". ".join(user_reviews[uid])
            num_toks = len(tokenizer(user_reviews[uid], truncation=False)['input_ids'])
            if num_toks >= min_user_review_len:
                keep_users[uid] = num_toks
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


def get_results(prediction_qid, ground_truth_qid, ranking_metrics, thresholds, ratios,
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

    qid_groups = defaultdict(set)
    for qid in ground_truth_qid.keys():
        for g in user_groups:
            if qid[:qid.index("_")] in user_groups[g]:
                qid_groups[g].add(qid)
                break

    if len(filtered_train_user_count) == 0:
        return
    print(f"grouped users in {time.time()-start}")

    # apply filters to the gt and pd
    ground_truth_qid = {qid: v for qid, v in ground_truth_qid.items() if qid[:qid.index("_")] in filtered_train_user_count}
    prediction_qid = {qid: v for qid, v in prediction_qid.items() if qid[:qid.index("_")] in filtered_train_user_count}

    # only count the number of users and interactions in this given eval file (for example when changes such as rev len ...)
    start = time.time()
    # users:
    num_filtered_eval_qids = len(ground_truth_qid.keys())
    gt_users = set([qid[:qid.index("_")] for qid in ground_truth_qid.keys()])
    num_filtered_eval_users = len(gt_users)

    num_filtered_eval_users_group = {group: 0 for group in user_groups}
    num_filtered_eval_qids_group = {group: 0 for group in user_groups}
    for g in user_groups:
        num_filtered_eval_qids_group[g] = len(set(ground_truth_qid.keys()).intersection(qid_groups[g]))
        num_filtered_eval_users_group[g] = len(gt_users.intersection(user_groups[g]))

    ret.append({"#total-qids": num_filtered_eval_qids})
    ret.append({"#total-users": num_filtered_eval_users})
    ret.extend([{f"#qids-{g}":  num_filtered_eval_qids_group[g]} for g in qid_groups])
    ret.extend([{f"#users-{g}":  num_filtered_eval_users_group[g]} for g in user_groups])

    # interactions:
    pos_total_cnt = 0
    neg_total_cnt = 0
    pos_inter_cnt = {group: 0 for group in user_groups}
    neg_inter_cnt = {group: 0 for group in user_groups}
    for qid in ground_truth_qid:
        pos_total_cnt += len([k for k, v in ground_truth_qid[qid].items() if v == 1])
        neg_total_cnt += len([k for k, v in ground_truth_qid[qid].items() if v == 0])

        group = None
        for g in qid_groups:
            if qid in qid_groups[g]:
                group = g
                break
        if group is not None:
            pos_inter_cnt[group] += len([k for k, v in ground_truth_qid[qid].items() if v == 1])
            neg_inter_cnt[group] += len([k for k, v in ground_truth_qid[qid].items() if v == 0])
    ret.append({"#total-pos-inter": pos_total_cnt})
    ret.append({"#total-neg-inter": neg_total_cnt})
    ret.extend([{f"#pos-inter-{g}": pos_inter_cnt[g]} for g in user_groups])
    ret.extend([{f"#neg-inter-{g}": neg_inter_cnt[g]} for g in user_groups])
    print(f"count inters {time.time() - start}")

    # calc metrics:
    # total:
    total_micro_results, total_macro_results = get_metrics(ground_truth=ground_truth_qid,
                                                           prediction_scores=prediction_qid,
                                                           ranking_metrics=ranking_metrics)
    metric_header = sorted(total_micro_results.keys())
    m_dict = {"ALL":  {f"micro_{h}": total_micro_results[h] for h in metric_header}}
    m_dict["ALL"].update({f"macro_{h}": total_macro_results[h] for h in metric_header})
    ret.append(m_dict)

    # groups:
    for g in user_groups:
        group_micro_results, group_macro_results = get_metrics(
            ground_truth={k: v for k, v in ground_truth_qid.items() if k in qid_groups[g]},
            prediction_scores={k: v for k, v in prediction_qid.items() if k in qid_groups[g]},
            ranking_metrics=ranking_metrics)
        m_dict = {g:  {f"micro_{h}": group_micro_results[h] for h in metric_header}}
        m_dict[g].update({f"macro_{h}": group_macro_results[h] for h in metric_header})
        ret.append(m_dict)

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

    prediction_raw = json.load(open(args.pred_path))
    if len(prediction_raw.keys()) == 1 and "predicted" in prediction_raw:
        prediction_raw = prediction_raw["predicted"]
    pos_file = pd.read_csv(args.pos_gt_path, dtype=str)
    neg_file = pd.read_csv(args.neg_gt_path, dtype=str)
    ground_truth_ = defaultdict(lambda: defaultdict())
    prediction_ = defaultdict(lambda: defaultdict())
    for user_id, item_id in zip(pos_file["user_id"], pos_file["item_id"]):
        ground_truth_[f"{user_id}_{item_id}"][item_id] = 1  # TODO if we are doing rating or something else change this
        prediction_[f"{user_id}_{item_id}"][item_id] = prediction_raw[user_id][item_id]
    for user_id, item_id, ref_item in zip(neg_file["user_id"], neg_file["item_id"], neg_file["ref_item"]):
        ground_truth_[f"{user_id}_{ref_item}"][item_id] = 0  # TODO if we are doing rating or something else change this
        prediction_[f"{user_id}_{ref_item}"][item_id] = prediction_raw[user_id][item_id]

    results = get_results(prediction_, ground_truth_, ranking_metrics, thrs, rates,
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