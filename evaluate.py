## group users and items based on their count in training data and evaluate all ALL, and COLD, HOT separately.
# what is saved?
# dict user->item->score
# create different things to use metric functions
import argparse
import csv
import json
import os
import time
from collections import Counter

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.sql.functions import user

from SBR.utils.metrics import calculate_ranking_metrics, calculate_cl_micro, calculate_cl_macro
from SBR.utils.data_loading import load_crawled_goodreads_dataset

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


def get_metrics(ground_truth, prediction_scores, calc_cl_metrics=True):
    if len(ground_truth) == 0:
        return {}
    start = time.time()
    results = calculate_ranking_metrics(gt=ground_truth, pd=prediction_scores, relevance_level=relevance_level)
    print(f"ranking metrics in {time.time() - start}")
    if calc_cl_metrics:
        start = time.time()
        user_gt = {}
        user_pd = {}
        all_gt = []
        all_pd = []
        for u in ground_truth.keys():
            user_gt[u] = []
            user_pd[u] = []
            sorted_items = sorted(ground_truth[u].keys())
            user_gt[u] = [ground_truth[u][i] for i in sorted_items]
            all_gt.extend(user_gt[u])
            user_pd[u] = list((np.array([prediction_scores[u][i] for i in sorted_items]) > prediction_threshold).astype(int))
            all_pd.extend(user_pd[u])
        temp = calculate_cl_micro(ground_truth=all_gt, predictions=all_pd)
        results.update(temp)
        temp = calculate_cl_macro(gt_user=user_gt, pd_user=user_pd)
        results.update(temp)
        print(f"cl metrics in {time.time() - start}")
    return results


def group_users(config, thresholds):
    # here we have some users who only exist in training set
    sp_files = {"train": os.path.join(config['dataset']['dataset_path'], "train.csv"),
                "validation": os.path.join(config['dataset']['dataset_path'], "validation.csv"),
                "test": os.path.join(config['dataset']['dataset_path'], "test.csv")}
    split_datasets = load_dataset("csv", data_files=sp_files)
    split_datasets = split_datasets.map(lambda x: {'rating': goodreads_rating_mapping[x['rating']]})
    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)

    train_user_count = Counter(split_datasets['train']['user_id'])
    eval_users = set(split_datasets['test']['user_id'])
    eval_users.update(set(split_datasets['validation']['user_id']))
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
    return ret_group, train_user_count, train_user_count_longtail


def main(config, valid_gt, valid_pd, test_gt, test_pd, thresholds, outf, valid_csv_f, test_csv_f):
    start = time.time()
    user_groups, train_user_count, train_user_count_longtail = group_users(config, thresholds)
    print(f"grouped users in {time.time()-start}")

    assert sum([len(ug) for ug in user_groups.values()]) == len(set(test_gt.keys()).union(valid_gt.keys()))

    # let's count how many interactions are there
    valid_pos_inter_cnt = {group: 0 for group in user_groups}
    valid_neg_inter_cnt = {group: 0 for group in user_groups}
    for u in valid_gt:
        group = [k for k in user_groups if u in user_groups[k]][0]
        valid_pos_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 1])
        valid_neg_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 0])

    test_pos_inter_cnt = {group: 0 for group in user_groups}
    test_neg_inter_cnt = {group: 0 for group in user_groups}
    for u in test_gt:
        group = [k for k in user_groups if u in user_groups[k]][0]
        test_pos_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 1])
        test_neg_inter_cnt[group] += len([k for k, v in valid_gt[u].items() if v == 0])

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
    rows_valid.append(["group"] + csv_metric_header)
    rows_test.append(["group"] + csv_metric_header)
    # ALL:
    valid_results = get_metrics(ground_truth=valid_gt,
                                prediction_scores=valid_pd,
                                calc_cl_metrics=calc_cl_metric)
    outf.write(f"Valid results ALL: {valid_results}\n")
    rows_valid.append(["Valid - ALL"] + [valid_results[h] for h in csv_metric_header])

    test_results = get_metrics(ground_truth=test_gt,
                               prediction_scores=test_pd,
                               calc_cl_metrics=calc_cl_metric)
    outf.write(f"Test results ALL: {test_results}\n\n")
    rows_test.append(["Test - ALL"] + [test_results[h] for h in csv_metric_header])

    # GROUPS
    for gr in user_groups:
        valid_results = get_metrics(ground_truth={k: v for k, v in valid_gt.items() if k in user_groups[gr]},
                                    prediction_scores={k: v for k, v in valid_pd.items() if k in user_groups[gr]},
                                    calc_cl_metrics=calc_cl_metric)
        outf.write(f"Valid results group: {gr}: {valid_results}\n")
        rows_valid.append([f"Valid - group {gr}"] + [valid_results[h] if h in valid_results else "" for h in csv_metric_header])

        test_results = get_metrics(ground_truth={k: v for k, v in test_gt.items() if k in user_groups[gr]},
                                   prediction_scores={k: v for k, v in test_pd.items() if k in user_groups[gr]},
                                   calc_cl_metrics=calc_cl_metric)
        outf.write(f"Test results group: {gr}: {test_results}\n\n")
        rows_test.append([f"Test - group {gr}"] + [test_results[h] if h in test_results else "" for h in csv_metric_header])

    vwriter = csv.writer(valid_csv_f)
    vwriter.writerows(rows_valid)
    twriter = csv.writer(test_csv_f)
    twriter.writerows(rows_test)


if __name__ == "__main__":
    # hard coded
    calc_cl_metric = False
    csv_metric_header = ["P_1", "Rprec", "ndcg_cut_5", "ndcg_cut_10", "ndcg_cut_20"]

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--thresholds', type=int, nargs='+', default=None, help='user thresholds')
    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
    thrs = args.thresholds

    if not os.path.exists(os.path.join(result_folder, "config.json")):
        raise ValueError(f"Result file config.json does not exist: {result_folder}")
    config = json.load(open(os.path.join(result_folder, "config.json")))

    if not os.path.exists(os.path.join(result_folder, "test_predicted.json")):
        raise ValueError(f"Result file test_output.json does not exist: {result_folder}")
    valid_ground_truth = json.load(open(os.path.join(result_folder, "best_valid_ground_truth.json")))
    valid_prediction = json.load(open(os.path.join(result_folder, "best_valid_predicted.json")))
    test_ground_truth = json.load(open(os.path.join(result_folder, "test_ground_truth.json")))
    test_prediction = json.load(open(os.path.join(result_folder, "test_predicted.json")))

    outfile_name = os.path.join(result_folder, f"results_th_{'_'.join([str(t) for t in thrs])}.txt")
    print(outfile_name)
    outfile = open(outfile_name, 'w')

    valid_csv = open(os.path.join(result_folder, f"results_valid_th_{'_'.join([str(t) for t in thrs])}.csv"), "w")
    test_csv = open(os.path.join(result_folder, f"results_test_th_{'_'.join([str(t) for t in thrs])}.csv"), "w")

    main(config, valid_ground_truth['ground_truth'], valid_prediction['predicted'],
         test_ground_truth['ground_truth'], test_prediction['predicted'],
         thrs, outfile, valid_csv, test_csv)

    outfile.close()
    valid_csv.close()
    test_csv.close()
