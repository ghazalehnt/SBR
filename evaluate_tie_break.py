## group users and items based on their count in training data and evaluate all ALL, and COLD, HOT separately.
# what is saved?
# dict user->item->score
# create different things to use metric functions
import argparse
import csv
import json
import os
import pickle
import time
from collections import Counter, defaultdict

import transformers
import pandas as pd

from SBR.utils.metrics import calculate_ranking_metrics_macro_avg_over_users

relevance_level = 1
prediction_threshold = 0.5

goodreads_rating_mapping = {
    None: None,  ## this means there was no rating
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def get_metrics(ground_truth, prediction_scores, weighted_labels, ranking_metrics=None):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics_macro_avg_over_users(gt=ground_truth, pd=prediction_scores,
                                                             relevance_level=relevance_level,
                                                             given_ranking_metrics=ranking_metrics,
                                                             calc_pytrec=not weighted_labels)
    print(f"ranking metrics in {time.time() - start}")
    return results


def group_users(config, thresholds, min_user_review_len=None, review_field=None):
    # here we have some users who only exist in training set
    split_datasets = defaultdict()
    for sp in ["train", "validation", "test"]:
        split_datasets[sp] = pd.read_csv(os.path.join(config['dataset']['dataset_path'], f"{sp}.csv"), dtype=str)

        if config['dataset']["name"] == "CGR":
            for k, v in goodreads_rating_mapping.items():
                split_datasets[sp]['rating'] = split_datasets[sp]['rating'].replace(k, v)
        elif config['dataset']["name"] == "GR_UCSD":
            split_datasets[sp]['rating'] = split_datasets[sp]['rating'].astype(int)
        elif config['dataset']["name"] == "Amazon":
            split_datasets[sp]['rating'] = split_datasets[sp]['rating'].astype(float).astype(int)
        else:
            raise NotImplementedError(f"dataset {config['dataset']['name']} not implemented!")

    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)

    train_user_count = Counter(split_datasets['train']['user_id'])

    # here we have users with long reviews and rest would neet to be intersect with it
    if min_user_review_len is not None:
        # don't calc for limited train data as they are not comparable at this point
        if 'limit_training_data' in config['dataset'] and config['dataset']['limit_training_data'] != "":
            return {}, {}, set()
        
        split_datasets['train'][review_field] = split_datasets['train'][review_field].fillna("")
        keep_users = {}
        tokenizer = transformers.AutoTokenizer.from_pretrained(BERTMODEL)
        user_reviews = {}
        for user_id, review in zip(split_datasets['train']['user_id'], split_datasets['train'][review_field]):
            if user_id not in user_reviews:
                user_reviews[user_id] = []
            if review is not None:
                user_reviews[user_id].append(review)
        for user_id in user_reviews:
            user_reviews[user_id] = ". ".join(user_reviews[user_id])
            num_toks = len(tokenizer(user_reviews[user_id], truncation=False)['input_ids'])
            if num_toks >= min_user_review_len:
                keep_users[user_id] = num_toks
        # print(keep_users)
        train_user_count = {k: v for k, v in train_user_count.items() if k in keep_users.keys()}

    eval_users = set(split_datasets['test']['user_id'])
    eval_users.update(set(split_datasets['validation']['user_id']))
    eval_users = eval_users.intersection(train_user_count.keys())
    train_user_count_longtail = {str(k): v for k, v in train_user_count.items() if k not in eval_users}

    groups = {thr: set() for thr in sorted(thresholds)}
    groups['rest'] = set()
    for user in eval_users:
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

    train_user_count = {str(k): v for k, v in train_user_count.items()}
    return ret_group, train_user_count, train_user_count_longtail


def jaccard_index(X, Y):
    X = set(X)
    return len(X.intersection(Y))/len(X.union(Y))


def main(config, valid_gt, valid_pd, test_gt, test_pd, thresholds,
         min_user_review_len=None, review_field=None, test_neg_st=None, valid_neg_st=None, ranking_metrics=None):
    start = time.time()
    user_groups, train_user_count, train_user_count_longtail = group_users(config, thresholds,
                                                                           min_user_review_len, review_field)
    if len(train_user_count) == 0:
        return

    print(f"grouped users in {time.time()-start}")

    if min_user_review_len is not None:
        outfile_name = os.path.join(result_folder,
                                    f"results_th_{'_'.join([str(t) for t in thrs])}_min_review_len_{min_user_review_len}_v-{valid_neg_st}_t-{test_neg_st}_tiebreak.txt")
        valid_csv_f = open(os.path.join(result_folder,
                                      f"results_valid_th_{'_'.join([str(t) for t in thrs])}_min_review_len_{min_user_review_len}_{valid_neg_st}_tiebreak.csv"), "w")
        test_csv_f = open(os.path.join(result_folder,
                                     f"results_test_th_{'_'.join([str(t) for t in thrs])}_min_review_len_{min_user_review_len}_{test_neg_st}_tiebreak.csv"), "w")
    else:
        outfile_name = os.path.join(result_folder, f"results_th_{'_'.join([str(t) for t in thrs])}_v-{valid_neg_st}_t-{test_neg_st}_tiebreak.txt")
        valid_csv_f = open(os.path.join(result_folder, f"results_valid_th_{'_'.join([str(t) for t in thrs])}_{valid_neg_st}_tiebreak.csv"), "w")
        test_csv_f = open(os.path.join(result_folder, f"results_test_th_{'_'.join([str(t) for t in thrs])}_{test_neg_st}_tiebreak.csv"), "w")

    print(outfile_name)
    outf = open(outfile_name, 'w')

    test_gt = {k: v for k, v in test_gt.items() if k in train_user_count.keys()}
    test_pd = {k: v for k, v in test_pd.items() if k in train_user_count.keys()}
    valid_gt = {k: v for k, v in valid_gt.items() if k in train_user_count.keys()}
    valid_pd = {k: v for k, v in valid_pd.items() if k in train_user_count.keys()}

    assert sum([len(ug) for ug in user_groups.values()]) == len(set(test_gt.keys()).union(valid_gt.keys()))

    # let's count how many interactions are there
    valid_pos_inter_cnt = {group: 0 for group in user_groups}
    valid_neg_inter_cnt = {group: 0 for group in user_groups}
    for u in valid_gt:
        group = [k for k in user_groups if u in user_groups[k]]
        if len(group) == 0:
            continue
        group = group[0]
        valid_pos_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 1])
        valid_neg_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 0])

    test_pos_inter_cnt = {group: 0 for group in user_groups}
    test_neg_inter_cnt = {group: 0 for group in user_groups}
    for u in test_gt:
        group = [k for k in user_groups if u in user_groups[k]]
        if len(group) == 0:
            continue
        group = group[0]
        test_pos_inter_cnt[group] += len([k for k, v in test_gt[u].items() if v == 1])
        test_neg_inter_cnt[group] += len([k for k, v in test_gt[u].items() if v == 0])

    outf.write(f"#total_evaluation_users = {len(set(test_gt.keys()).union(valid_gt.keys()))} \n"
               f"#total_training_users = {len(set(test_gt.keys()).union(valid_gt.keys())) + len(train_user_count_longtail)} \n"
               f"#total_longtail_trainonly_users = {len(train_user_count_longtail)} \n")
    for gr in user_groups:
        outf.write(f"#eval_user_group_{gr}: {len(user_groups[gr].intersection(set(test_gt.keys()).union(valid_gt.keys())))}  ")
        outf.write(f"#valid_user_group_{gr}: {len(user_groups[gr].intersection(valid_gt.keys()))}  ")
        outf.write(f"#test_user_group_{gr}: {len(user_groups[gr].intersection(test_gt.keys()))}\n")

    outf.write(f"#total_positive_inters_validation = {sum(list(valid_pos_inter_cnt.values()))}\n")
    for gr in user_groups:
        outf.write(f"positive_inters_validation_user_group_{gr} = {valid_pos_inter_cnt[gr]}\n")
    outf.write(f"#total_negatove_inters_validation = {sum(list(valid_neg_inter_cnt.values()))}\n")
    for gr in user_groups:
        outf.write(f"negative_inters_validation_user_group_{gr} = {valid_neg_inter_cnt[gr]}\n")

    outf.write(f"#total_positive_inters_test = {sum(list(test_pos_inter_cnt.values()))}\n")
    for gr in user_groups:
        outf.write(f"positive_inters_test_user_group_{gr} = {test_pos_inter_cnt[gr]}\n")
    outf.write(f"#total_negatove_inters_test = {sum(list(test_neg_inter_cnt.values()))}\n")
    for gr in user_groups:
        outf.write(f"negative_inters_test_user_group_{gr} = {test_neg_inter_cnt[gr]}\n")
    outf.write("\n\n")

    rows_valid = []
    rows_test = []
    # ALL:
    valid_results = get_metrics(ground_truth=valid_gt,
                                prediction_scores=valid_pd,
                                weighted_labels=True if "-" in valid_neg_st else False,
                                ranking_metrics=ranking_metrics)
    metric_header = sorted(valid_results.keys())
    rows_valid.append(["group"] + metric_header)
    outf.write(f"Valid results ALL: {valid_results}\n")
    rows_valid.append(["Valid - ALL"] + [valid_results[h] for h in metric_header])

    test_results = get_metrics(ground_truth=test_gt,
                               prediction_scores=test_pd,
                               weighted_labels=True if "-" in test_neg_st else False,
                               ranking_metrics=ranking_metrics)
    rows_test.append(["group"] + metric_header)
    outf.write(f"Test results ALL: {test_results}\n\n")
    rows_test.append(["Test - ALL"] + [test_results[h] for h in metric_header])

    # GROUPS
    for gr in user_groups:
        valid_results = get_metrics(ground_truth={k: v for k, v in valid_gt.items() if k in user_groups[gr]},
                                    prediction_scores={k: v for k, v in valid_pd.items() if k in user_groups[gr]},
                                    weighted_labels=True if "-" in valid_neg_st else False,
                                    ranking_metrics=ranking_metrics)
        outf.write(f"Valid results group: {gr}: {valid_results}\n")
        rows_valid.append([f"Valid - group {gr}"] + [valid_results[h] if h in valid_results else "" for h in metric_header])

        test_results = get_metrics(ground_truth={k: v for k, v in test_gt.items() if k in user_groups[gr]},
                                   prediction_scores={k: v for k, v in test_pd.items() if k in user_groups[gr]},
                                   weighted_labels=True if "-" in test_neg_st else False,
                                   ranking_metrics=ranking_metrics)
        outf.write(f"Test results group: {gr}: {test_results}\n\n")
        rows_test.append([f"Test - group {gr}"] + [test_results[h] if h in test_results else "" for h in metric_header])

    vwriter = csv.writer(valid_csv_f)
    vwriter.writerows(rows_valid)
    twriter = csv.writer(test_csv_f)
    twriter.writerows(rows_test)

    outf.close()
    valid_csv_f.close()
    test_csv_f.close()


if __name__ == "__main__":
    # hard coded
    calc_cl_metric = False
    BERTMODEL = "bert-base-uncased"  # TODO hard coded

    parser = argparse.ArgumentParser()
    # basic:
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None, help='user thresholds')
    # required which evaluation set we want to evaluate, random or genre ?
    parser.add_argument('--test_neg_strategy', type=str, default="random_100", help='negative sampling strategy')
    parser.add_argument('--valid_neg_strategy', type=str, default="random_100", help='negative sampling strategy')

    # optional if we want to only calculate the metrics for users with certain review length.
    parser.add_argument('--user_review_len', type=int, default=None, help='min length of the user review')
    parser.add_argument('--review_field', type=str, default="review", help='review field')
    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
    thrs = args.thresholds

    r_len = args.user_review_len
    r_field = args.review_field

    test_neg_strategy = args.test_neg_strategy
    valid_neg_strategy = args.valid_neg_strategy

    if not os.path.exists(os.path.join(result_folder, "config.json")):
        raise ValueError(f"Result file config.json does not exist: {result_folder}")
    config = json.load(open(os.path.join(result_folder, "config.json")))

    test_prediction = json.load(open(os.path.join(result_folder,
                                                  f"test_predicted_test_neg_{test_neg_strategy}.json")))
    valid_prediction = json.load(open(os.path.join(result_folder,
                                                   f"best_valid_predicted_validation_neg_{valid_neg_strategy}.json")))
    test_ground_truth = json.load(open(os.path.join(result_folder,
                                                    f"test_ground_truth_test_neg_{test_neg_strategy}.json")))
    valid_ground_truth = json.load(open(os.path.join(result_folder,
                                                     f"best_valid_ground_truth_validation_neg_{valid_neg_strategy}.json")))

    epsilon = 1e-7
    for user in test_prediction['predicted'].keys():
        sorted_items = sorted(test_prediction['predicted'][user].keys(), reverse=True)
        for i in range(len(sorted_items)):
            for j in range(i+1, len(sorted_items)):
                if test_prediction['predicted'][user][sorted_items[i]] == test_prediction['predicted'][user][sorted_items[j]]:
                    test_prediction['predicted'][user][sorted_items[j]] += epsilon

    for user in valid_prediction['predicted'].keys():
        sorted_items = sorted(valid_prediction['predicted'][user].keys(), reverse=True)
        for i in range(len(sorted_items)):
            for j in range(i+1, len(sorted_items)):
                if valid_prediction['predicted'][user][sorted_items[i]] == valid_prediction['predicted'][user][sorted_items[j]]:
                    valid_prediction['predicted'][user][sorted_items[j]] += epsilon

    ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20", "P_1", "recip_rank"]
    if "-" in valid_neg_strategy or "-" in test_neg_strategy:
        ranking_metrics = ["ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"]

    main(config, valid_ground_truth['ground_truth'], valid_prediction['predicted'],
         test_ground_truth['ground_truth'], test_prediction['predicted'],
         thrs, r_len, r_field, test_neg_strategy, valid_neg_strategy, ranking_metrics)


# TODO test this script
