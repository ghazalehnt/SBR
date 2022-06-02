## group users and items based on their count in training data and evaluate all ALL, and COLD, HOT separately.
# what is saved?
# dict user->item->score
# create different things to use metric functions
import argparse
import json
import os
import time
from collections import Counter

from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np

from SBR.utils.metrics import calculate_ranking_metrics, calculate_cl_micro, calculate_cl_macro
from data_loading import load_crawled_goodreads_dataset

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


def visualize_train_set_interactions(config, cold_threshold):
    eval_cold_users, eval_warm_users, train_user_count, train_user_count_longtail, test_user_count, cold_test_inter_cnt, cold_valid_inter_cnt, warm_test_inter_cnt, warm_valid_inter_cnt = get_cold_users(config, cold_threshold)

    user_count_reverse_longtail = {i: [] for i in range(1, max(train_user_count.values()) + 1)}
    user_count_reverse = {i: [] for i in range(1, max(train_user_count.values()) + 1)}
    for u, v in train_user_count_longtail.items():
        user_count_reverse_longtail[v].append(u)
    for u, v in train_user_count.items():
        user_count_reverse[v].append(u)

    x = []
    y_all, y_longtail = {}, {}
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200, 300, 400, 500, 1000, 2185]
    for bi in range(1, len(bins)):
        y_all[bi] = 0
        y_longtail[bi] = 0
        x.append(str(bins[bi]))
        for c in user_count_reverse.keys():
            if bins[bi-1] < c <= bins[bi]:
                y_all[bi] += len(user_count_reverse[c])
                y_longtail[bi] += len(user_count_reverse_longtail[c])

    fig, ax = plt.subplots()
    ax.bar(x, y_all.values(), label='all')
    ax.bar(x, y_longtail.values(), label='only in train set')
    plt.ylabel("number of users")
    plt.xlabel("interactions")
    plt.legend()
    y = list(y_all.values())
    y2 = list(y_longtail.values())
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    for i in range(len(x)):
        if y2[i] > 0:
            plt.text(i, y2[i], y2[i])
    plt.show()

    cold_user_cound = {k:v for k, v in test_user_count.items() if str(k) in eval_cold_users}
    user_count_reverse = {i: [] for i in range(1, max(test_user_count.values()) + 1)}
    cold_user_count_reverse = {i: [] for i in range(1, max(test_user_count.values()) + 1)}
    for u, v in test_user_count.items():
        user_count_reverse[v].append(u)
    for u, v in cold_user_cound.items():
        cold_user_count_reverse[v].append(u)
    x = []
    y_all, y_cold = {}, {}
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 50, 100, 200, 300, 400, 500, 1000, 2185]
    for bi in range(1, len(bins)):
        y_all[bi] = 0
        y_cold[bi] = 0
        x.append(str(bins[bi]))
        for c in user_count_reverse.keys():
            if bins[bi - 1] < c <= bins[bi]:
                y_all[bi] += len(user_count_reverse[c])
                y_cold[bi] += len(cold_user_count_reverse[c])

    fig, ax = plt.subplots()
    ax.bar(x, y_all.values(), label='warm', color='red')
    ax.bar(x, y_cold.values(), label='cold', color='blue')
    plt.ylabel("number of users")
    plt.xlabel("interactions")
    plt.legend()
    y = list(y_all.values())
    y2 = list(y_cold.values())
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    for i in range(len(x)):
        if y2[i] > 0:
            plt.text(i, y2[i], y2[i])
    plt.show()


def get_metrics(ground_truth, prediction_scores):
    start = time.time()
    results = calculate_ranking_metrics(gt=ground_truth, pd=prediction_scores, relevance_level=relevance_level)
    print(f"ranking metrics in {time.time() - start}")
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


def get_cold_users(config, cold_threshold, with_text=False):
    # here we have some users who only exist in training set
    sp_files = {"train": os.path.join(config['dataset']['dataset_path'], "train.csv"),
                "validation": os.path.join(config['dataset']['dataset_path'], "validation.csv"),
                "test": os.path.join(config['dataset']['dataset_path'], "test.csv")}
    split_datasets = load_dataset("csv", data_files=sp_files)
    split_datasets = split_datasets.map(lambda x: {'rating': goodreads_rating_mapping[x['rating']]})
    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)

    # we did the training on the entire dataset, regardless of having text or not!
    # but we want to evaluate only for user and items with text...
    # in user case, with training-reviews, with item case description and genre.
    items_with_text = None
    if with_text:
        # load user and item info as well, and here we have hard coded user_text and item_text fields
        config['dataset']['user_text'] = user_text_fields
        config['dataset']['item_text'] = item_text_fields
        config['dataset']['user_review_choice'] = user_review_choice
        config['dataset']['text_in_batch'] = True
        _, user_info, item_info = load_crawled_goodreads_dataset(config['dataset'])
        user_info = user_info.to_pandas()
        users_with_text = set([str(i) for i in user_info[(user_info['text']) != '']['user_id']])
        item_info = item_info.to_pandas()
        dataset_item_cnt = len(set(item_info['item_id']))
        items_with_text = set([str(i) for i in item_info[(item_info['text']) != '']['item_id']])
        # filter eval users and items that don't have text:

    train_user_count = Counter(split_datasets['train']['user_id'])
    eval_users = set(split_datasets['test']['user_id'])
    eval_users.update(set(split_datasets['validation']['user_id']))
    train_user_count_longtail = {k: v for k, v in train_user_count.items() if k not in eval_users}

    cold_users = set()
    warm_users = set()
    for user in eval_users:
        if train_user_count[user] > cold_threshold:
            warm_users.add(str(user))
        else:
            cold_users.add(str(user))

    return cold_users, warm_users, train_user_count, train_user_count_longtail, users_with_text, items_with_text, dataset_item_cnt


def main(config, valid_gt, valid_pd, test_gt, test_pd, cold_threshold, with_text, outf):
    start = time.time()
    eval_cold_users, eval_warm_users, train_user_count, train_user_count_longtail, users_with_text, items_with_text, dataset_item_cnt = \
        get_cold_users(config, cold_threshold, with_text)
    print(f"get cold/warm users in {time.time()-start}")

    # let's count how many interactions were there before with text
    before_valid_pos_inter_cnt_warm = 0
    before_valid_neg_inter_cnt_warm = 0
    before_valid_pos_inter_cnt_cold = 0
    before_valid_neg_inter_cnt_cold = 0
    for u in valid_gt:
        if u in eval_warm_users:
            before_valid_pos_inter_cnt_warm += len([k for k, v in valid_gt[u].items() if v == 1])
            before_valid_neg_inter_cnt_warm += len([k for k, v in valid_gt[u].items() if v == 0])
        elif u in eval_cold_users:
            before_valid_pos_inter_cnt_cold += len([k for k, v in valid_gt[u].items() if v == 1])
            before_valid_neg_inter_cnt_cold += len([k for k, v in valid_gt[u].items() if v == 0])
    before_test_pos_inter_cnt_warm = 0
    before_test_neg_inter_cnt_warm = 0
    before_test_pos_inter_cnt_cold = 0
    before_test_neg_inter_cnt_cold = 0
    for u in test_gt:
        if u in eval_warm_users:
            before_test_pos_inter_cnt_warm += len([k for k, v in test_gt[u].items() if v == 1])
            before_test_neg_inter_cnt_warm += len([k for k, v in test_gt[u].items() if v == 0])
        if u in eval_cold_users:
            before_test_pos_inter_cnt_cold += len([k for k, v in test_gt[u].items() if v == 1])
            before_test_neg_inter_cnt_cold += len([k for k, v in test_gt[u].items() if v == 0])

    # first filter users with text:
    valid_gt = {u: v for u, v in valid_gt.items() if u in users_with_text}
    valid_pd = {u: v for u, v in valid_pd.items() if u in users_with_text}
    test_gt = {u: v for u, v in test_gt.items() if u in users_with_text}
    test_pd = {u: v for u, v in test_pd.items() if u in users_with_text}

    # then we want to filter items with text now here:
    if with_text:
        for u in valid_gt:
            valid_gt[u] = {k: v for k, v in valid_gt[u].items() if k in items_with_text}
        for u in valid_pd:
            valid_pd[u] = {k: v for k, v in valid_pd[u].items() if k in items_with_text}
        for u in test_gt:
            test_gt[u] = {k: v for k, v in test_gt[u].items() if k in items_with_text}
        for u in test_pd:
            test_pd[u] = {k: v for k, v in test_pd[u].items() if k in items_with_text}
        # lets count how many are there now
        after_valid_pos_inter_cnt_warm = 0
        after_valid_neg_inter_cnt_warm = 0
        after_valid_pos_inter_cnt_cold = 0
        after_valid_neg_inter_cnt_cold = 0
        for u in valid_gt:
            if u in eval_warm_users:
                after_valid_pos_inter_cnt_warm += len([k for k, v in valid_gt[u].items() if v == 1])
                after_valid_neg_inter_cnt_warm += len([k for k, v in valid_gt[u].items() if v == 0])
            elif u in eval_cold_users:
                after_valid_pos_inter_cnt_cold += len([k for k, v in valid_gt[u].items() if v == 1])
                after_valid_neg_inter_cnt_cold += len([k for k, v in valid_gt[u].items() if v == 0])
        after_test_pos_inter_cnt_warm = 0
        after_test_neg_inter_cnt_warm = 0
        after_test_pos_inter_cnt_cold = 0
        after_test_neg_inter_cnt_cold = 0
        for u in test_gt:
            if u in eval_warm_users:
                after_test_pos_inter_cnt_warm += len([k for k, v in test_gt[u].items() if v == 1])
                after_test_neg_inter_cnt_warm += len([k for k, v in test_gt[u].items() if v == 0])
            if u in eval_cold_users:
                after_test_pos_inter_cnt_cold += len([k for k, v in test_gt[u].items() if v == 1])
                after_test_neg_inter_cnt_cold += len([k for k, v in test_gt[u].items() if v == 0])

    outf.write(f"BEFORE-TEXT-FILTER #total_evaluation_users = {len(eval_cold_users) + len(eval_warm_users)} \n"
               f"BEFORE-TEXT-FILTER #total_training_users = {len(eval_cold_users) + len(eval_warm_users) + len(train_user_count_longtail)} \n"
               f"BEFORE-TEXT-FILTER #total_longtail_trainonly_users = {len(train_user_count_longtail)} \n")
    outf.write(f"BEFORE-TEXT-FILTER #eval_users_warm = {len(eval_warm_users)}\n"
               f"BEFORE-TEXT-FILTER #eval_users_cold = {len(eval_cold_users)} (interactions <= {cold_threshold})\n\n")

    outf.write(f"BEFORE-TEXT-FILTER #total_positive_inters_validation = "
               f"{before_valid_pos_inter_cnt_warm+before_valid_pos_inter_cnt_cold} "
               f"(WARM = {before_valid_pos_inter_cnt_warm}, COLD = {before_valid_pos_inter_cnt_cold})\n"
               f"BEFORE-TEXT-FILTER #total_negative_inters_validation = "
               f"{before_valid_neg_inter_cnt_warm+before_valid_neg_inter_cnt_cold} "
               f"(WARM = {before_valid_neg_inter_cnt_warm}, COLD = {before_valid_neg_inter_cnt_cold})\n"
               f"BEFORE-TEXT-FILTER #total_positive_inters_test = "
               f"{before_test_pos_inter_cnt_warm+before_test_pos_inter_cnt_cold} "
               f"(WARM = {before_test_pos_inter_cnt_warm}, COLD = {before_test_pos_inter_cnt_cold})\n"
               f"BEFORE-TEXT-FILTER #total_negative_inters_test = "
               f"{before_test_neg_inter_cnt_warm + before_test_neg_inter_cnt_cold} "
               f"(WARM = {before_test_neg_inter_cnt_warm}, COLD = {before_test_neg_inter_cnt_cold})\n\n")

    if with_text:
        outf.write(f"AFTER-TEXT-FILTER #total_evaluation_users = {len(eval_cold_users.intersection(users_with_text)) + len(eval_warm_users.intersection(users_with_text))} \n")
        outf.write(f"AFTER-TEXT-FILTER #eval_users_warm = {len(eval_warm_users.intersection(users_with_text))}\n"
                   f"AFTER-TEXT-FILTER #eval_users_cold = {len(eval_cold_users.intersection(users_with_text))} (interactions <= {cold_threshold})\n\n")

        outf.write(f"total dataset items with text = {len(items_with_text)}/{dataset_item_cnt}\n")
        outf.write(f"AFTER-TEXT-FILTER #total_positive_inters_validation = "
                   f"{after_valid_pos_inter_cnt_warm + after_valid_pos_inter_cnt_cold} "
                   f"(WARM = {after_valid_pos_inter_cnt_warm}, COLD = {after_valid_pos_inter_cnt_cold})\n"
                   f"AFTER-TEXT-FILTER #total_negative_inters_validation = "
                   f"{after_valid_neg_inter_cnt_warm + after_valid_neg_inter_cnt_cold} "
                   f"(WARM = {after_valid_neg_inter_cnt_warm}, COLD = {after_valid_neg_inter_cnt_cold})\n"
                   f"AFTER-TEXT-FILTER #total_positive_inters_test = "
                   f"{after_test_pos_inter_cnt_warm + after_test_pos_inter_cnt_cold} "
                   f"(WARM = {after_test_pos_inter_cnt_warm}, COLD = {after_test_pos_inter_cnt_cold})\n"
                   f"AFTER-TEXT-FILTER #total_negative_inters_test = "
                   f"{after_test_neg_inter_cnt_warm + after_test_neg_inter_cnt_cold} "
                   f"(WARM = {after_test_neg_inter_cnt_warm}, COLD = {after_test_neg_inter_cnt_cold})\n\n")

    outf.write(f"#validation_users_warm = {len(eval_warm_users.intersection(valid_gt.keys()))}\n"
               f"#validation_users_cold = {len(eval_cold_users.intersection(valid_gt.keys()))} (interactions <= {cold_threshold})\n\n")
    outf.write(f"#test_users_warm = {len(eval_warm_users.intersection(test_gt.keys()))}\n"
               f"#test_users_cold = {len(eval_cold_users.intersection(test_gt.keys()))} (interactions <= {cold_threshold})\n\n")

    # ALL:
    valid_results = get_metrics(ground_truth=valid_gt,
                                prediction_scores=valid_pd)
    outf.write(f"Valid results ALL: {valid_results}\n")

    test_results = get_metrics(ground_truth=test_gt,
                               prediction_scores=test_pd)
    outf.write(f"Test results ALL: {test_results}\n\n")

    # WARM:
    valid_results = get_metrics(ground_truth={k: v for k, v in valid_gt.items() if k in eval_warm_users},
                                prediction_scores={k: v for k, v in valid_pd.items() if k in eval_warm_users})
    outf.write(f"Valid results WARM (> {cold_threshold}): {valid_results}\n")

    test_results = get_metrics(ground_truth={k: v for k, v in test_gt.items() if k in eval_warm_users},
                               prediction_scores={k: v for k, v in test_pd.items() if k in eval_warm_users})
    outf.write(f"Test results WARM (> {cold_threshold}): {test_results}\n\n")

    # COLD:
    valid_results = get_metrics(ground_truth={k: v for k, v in valid_gt.items() if k in eval_cold_users},
                                prediction_scores={k: v for k, v in valid_pd.items() if k in eval_cold_users})
    outf.write(f"Valid results COLD (<= {cold_threshold}): {valid_results}\n")

    test_results = get_metrics(ground_truth={k: v for k, v in test_gt.items() if k in eval_cold_users},
                               prediction_scores={k: v for k, v in test_pd.items() if k in eval_cold_users})
    outf.write(f"Test results COLD (<= {cold_threshold}): {test_results}\n\n")


if __name__ == "__main__":
    user_text_fields = ['interaction.review']
    user_review_choice = "pos_rating_sorted_3"
    item_text_fields = ['item.genres', 'item.description']  # here 'item.title' is removed, becasue we want to see which items have description or genre and title does not matter

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--vis', type=bool, default=False, help='want to see the train interactions')
    parser.add_argument('--cold_th', type=int, default=None, help='cold user threshold')
    parser.add_argument('--t', type=bool, default=False, help='evaluate only the users and items that have text')
    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
    if not os.path.exists(os.path.join(result_folder, "config.json")):
        raise ValueError(f"Result file config.json does not exist: {result_folder}")
    config = json.load(open(os.path.join(result_folder, "config.json")))
    if args.vis:
        visualize_train_set_interactions(config, args.cold_th)
    else:
        if not os.path.exists(os.path.join(result_folder, "test_predicted.json")):
            raise ValueError(f"Result file test_output.json does not exist: {result_folder}")
        valid_ground_truth = json.load(open(os.path.join(result_folder, "best_valid_ground_truth.json")))
        valid_prediction = json.load(open(os.path.join(result_folder, "best_valid_predicted.json")))
        test_ground_truth = json.load(open(os.path.join(result_folder, "test_ground_truth.json")))
        test_prediction = json.load(open(os.path.join(result_folder, "test_predicted.json")))
        print(os.path.join(result_folder, f"results_coldth{args.cold_th}_withtext{args.t}.txt"))
        outfile = open(os.path.join(result_folder, f"results_coldth{args.cold_th}_withtext{args.t}.txt"), 'w')
        main(config, valid_ground_truth['ground_truth'], valid_prediction['predicted'],
             test_ground_truth['ground_truth'], test_prediction['predicted'],
             args.cold_th, args.t, outfile)
        outfile.close()
