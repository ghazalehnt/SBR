## group users and items based on their count in training data and evaluate all ALL, and COLD, HOT separately.
# what is saved?
# dict user->item->score
# create different things to use metric functions
import argparse
import json
import os
from collections import Counter

from datasets import load_dataset
import matplotlib.pyplot as plt

from metrics import calculate_ranking_metrics, calculate_cl_micro, calculate_cl_macro

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


def visualize_train_set_interactions(config):
    # here we have some users who only exist in training set
    sp_files = {"train": os.path.join(config['dataset']['dataset_path'], "train.csv"),
                "validation": os.path.join(config['dataset']['dataset_path'], "validation.csv"),
                "test": os.path.join(config['dataset']['dataset_path'], "test.csv")}
    split_datasets = load_dataset("csv", data_files=sp_files)
    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.map(lambda x: {'rating': goodreads_rating_mapping[x['rating']]})
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)
    train_user_count = Counter(split_datasets['train']['user_id'])
    test_users = set(split_datasets['test']['user_id'])
    test_users.update(set(split_datasets['validation']['user_id']))

    # we want to show which users are only in train set (named lt)
    train_user_count_longtail = {k: v for k, v in train_user_count.items() if k not in test_users}
    # train_user_count_eval = {k: v for k, v in train_user_count.items() if k in test_users}

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


def get_metrics(ground_truth, prediction_scores):
    results = calculate_ranking_metrics(gt=ground_truth, pd=prediction_scores, relevance_level=relevance_level)
    user_gt = {}
    user_pd = {}
    all_gt = []
    all_pd = []
    for u in ground_truth.keys():
        user_gt[u] = []
        user_pd[u] = []
        u_items = ground_truth[u].keys()
        for item in u_items:
            user_gt[u].append(ground_truth[u][item])
            all_gt.append(ground_truth[u][item])
            user_pd[u].append(1 if prediction_scores[u][item] > prediction_threshold else 0)
            all_pd.append(1 if prediction_scores[u][item] > prediction_threshold else 0)
    temp = calculate_cl_micro(ground_truth=all_gt, predictions=all_pd)
    results.update(temp)
    temp = calculate_cl_macro(gt_user=user_gt, pd_user=user_pd)
    results.update(temp)
    return results


def get_cold_users(config, cold_threshold):
    # here we have some users who only exist in training set
    sp_files = {"train": os.path.join(config['dataset']['dataset_path'], "train.csv"),
                "validation": os.path.join(config['dataset']['dataset_path'], "validation.csv"),
                "test": os.path.join(config['dataset']['dataset_path'], "test.csv")}
    split_datasets = load_dataset("csv", data_files=sp_files)
    if not config['dataset']['binary_interactions']:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.map(lambda x: {'rating': goodreads_rating_mapping[x['rating']]})
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)
    train_user_count = Counter(split_datasets['train']['user_id'])
    test_users = set(split_datasets['test']['user_id'])
    test_users.update(set(split_datasets['validation']['user_id']))

    cold_users = set()
    warm_users = set()
    for user in test_users:
        if train_user_count[user] > cold_threshold:
            warm_users.add(str(user))
        else:
            cold_users.add(str(user))
    return cold_users, warm_users


def main(config, valid_output, test_output, cold_threshold):
    cold_users, warm_users = get_cold_users(config, cold_threshold)

    # ALL:
    valid_results = get_metrics(ground_truth=valid_output["ground_truth"], prediction_scores=valid_output["predicted"])
    print(f"Valid results (ALL): {valid_results}")

    test_results = get_metrics(ground_truth=test_output["ground_truth"], prediction_scores=test_output["predicted"])
    print(f"Test results (ALL): {test_results}")

    # WARM:
    valid_results = get_metrics(ground_truth={k: v for k, v in valid_output["ground_truth"].items() if k in warm_users},
                                prediction_scores={k: v for k, v in valid_output["predicted"].items() if k in warm_users})
    print(f"Valid results (WARM): {valid_results}")

    test_results = get_metrics(ground_truth={k: v for k, v in test_output["ground_truth"].items() if k in warm_users},
                               prediction_scores={k: v for k, v in test_output["predicted"].items() if k in warm_users})
    print(f"Test results (WARM): {test_results}")

    # COLD:
    valid_results = get_metrics(ground_truth={k: v for k, v in valid_output["ground_truth"].items() if k in cold_users},
                                prediction_scores={k: v for k, v in valid_output["predicted"].items() if k in cold_users})
    print(f"Valid results (COLD): {valid_results}")

    test_results = get_metrics(ground_truth={k: v for k, v in test_output["ground_truth"].items() if k in cold_users},
                               prediction_scores={k: v for k, v in test_output["predicted"].items() if k in cold_users})
    print(f"Test results (COLD): {test_results}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_folder', '-r', type=str, default=None, help='result folder, to evaluate')
    parser.add_argument('--vis', type=bool, default=False, help='want to see the train interactions')
    parser.add_argument('--cold_th', type=int, default=None, help='cold user threshold')
    args, _ = parser.parse_known_args()

    result_folder = args.result_folder
    if not os.path.exists(os.path.join(result_folder, "config.json")):
        raise ValueError(f"Result file config.json does not exist: {result_folder}")
    config = json.load(open(os.path.join(result_folder, "config.json")))
    if args.vis:
        visualize_train_set_interactions(config)
    else:
        if not os.path.exists(os.path.join(result_folder, "test_output.json")):
            raise ValueError(f"Result file test_output.json does not exist: {result_folder}")
        valid_output = json.load(open(os.path.join(result_folder, "best_valid_output.json")))
        test_output = json.load(open(os.path.join(result_folder, "test_output.json")))
        main(config, valid_output, test_output, args.cold_th)