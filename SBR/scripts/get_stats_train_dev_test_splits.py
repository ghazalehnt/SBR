import argparse
import csv
import math
from collections import Counter
from os.path import join

import pandas as pd

import networkx as nx
from networkx.algorithms import bipartite
import scipy.stats
import matplotlib.pyplot as plt


def get_histogram(vals):
    hist = {"1-5": 0, "6-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "61-70": 0, "71-80": 0,
              "81-90": 0, "91-100": 0, "101-200": 0, "201-500": 0, "501+": 0}
    for v in vals:
        if 1 <= v <= 5:
            hist["1-5"] += 1
        elif 6 <= v <= 10:
            hist["6-10"] += 1
        elif 11 <= v <= 20:
            hist["11-20"] += 1
        elif 21 <= v <= 30:
            hist["21-30"] += 1
        elif 31 <= v <= 40:
            hist["31-40"] += 1
        elif 41 <= v <= 50:
            hist["41-50"] += 1
        elif 51 <= v <= 60:
            hist["51-60"] += 1
        elif 61 <= v <= 70:
            hist["61-70"] += 1
        elif 71 <= v <= 80:
            hist["71-80"] += 1
        elif 81 <= v <= 90:
            hist["81-90"] += 1
        elif 91 <= v <= 100:
            hist["91-100"] += 1
        elif 101 <= v <= 200:
            hist["101-200"] += 1
        elif 201 <= v <= 500:
            hist["201-500"] += 1
        elif v >= 501:
            hist["501+"] += 1
    return hist


def read_interactions(dataset_path):
    ret = {}
    sp_files = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    for split, sp_file in sp_files.items():
        ret[split] = pd.read_csv(join(dataset_path, sp_file), dtype=str)
    return ret["train"], ret["validation"], ret["test"]


def read_item_info(dataset_path):
    return pd.read_csv(join(dataset_path, "items.csv"), dtype=str)


def get_graph(inters):
    B = nx.Graph()
    user_nodes = set()
    item_nodes = set()
    edges = []
    for u, i in zip(inters[USER_ID_FIELD], inters[ITEM_ID_FIELD]):
        user_id = f"u_{u}"
        item_id = f"i_{i}"
        user_nodes.add(user_id)
        item_nodes.add(item_id)
        edges.append((user_id, item_id))
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(list(user_nodes), bipartite=0)
    B.add_nodes_from(list(item_nodes), bipartite=1)
    # Add edges only between nodes of opposite node sets
    B.add_edges_from(edges)
    return B, user_nodes


def get_user_groups(train_set, FIELD, thresholds=[5]):
    user_count = Counter(train_set[FIELD])
    groups = {thr: set() for thr in sorted(thresholds)}
    groups['rest'] = set()
    for user, cnt in user_count.items():
        added = False
        for thr in sorted(thresholds):
            if cnt <= thr:
                groups[thr].add(user)
                added = True
                break
        if not added:
            groups['rest'].add(user)

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


def user_grp_inter_cnt(split_set, users, field):
    return len(split_set[split_set[field].isin(users)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default=None, help='path to dataset')
    parser.add_argument('--af', type=str, default=None, help='AUTHOR_FIELD')
    args, _ = parser.parse_known_args()

    DATASET_DIR = args.dataset_path
    AUTHOR_FIELD = args.af

    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "item_id"
    thresholds = [5, 50]
    statfile = open(join(DATASET_DIR, f"stats_{'-'.join([str(thr) for thr in thresholds])}.txt"), 'w')

    train, valid, test = read_interactions(DATASET_DIR)
    item_info = read_item_info(DATASET_DIR)
    train = train.merge(item_info[[ITEM_ID_FIELD, AUTHOR_FIELD]], "left", on=ITEM_ID_FIELD)
    valid = valid.merge(item_info[[ITEM_ID_FIELD, AUTHOR_FIELD]], "left", on=ITEM_ID_FIELD)
    test = test.merge(item_info[[ITEM_ID_FIELD, AUTHOR_FIELD]], "left", on=ITEM_ID_FIELD)

    user_groups = get_user_groups(train, USER_ID_FIELD, thresholds)
    test_users = set(test[USER_ID_FIELD])
    valid_users = set(valid[USER_ID_FIELD])

    item_groups = get_user_groups(train, ITEM_ID_FIELD, thresholds)
    train_items = set(train[ITEM_ID_FIELD])
    test_items = set(test[ITEM_ID_FIELD])
    valid_items = set(valid[ITEM_ID_FIELD])

    per_user = {"train": Counter(train[USER_ID_FIELD]),
                "valid": Counter(valid[USER_ID_FIELD]),
                "test":  Counter(test[USER_ID_FIELD])}

    per_item = {"train": Counter(train[ITEM_ID_FIELD]),
                "valid": Counter(valid[ITEM_ID_FIELD]),
                "test": Counter(test[ITEM_ID_FIELD])}

    all_interactions_train = sum(per_user['train'].values())
    all_interactions_valid = sum(per_user['valid'].values())
    all_interactions_test = sum(per_user['test'].values())
    total_interactions = all_interactions_train + all_interactions_valid + all_interactions_test

    h_user = get_histogram(per_user['train'].values())
    x = list(h_user.keys())
    y = list(h_user.values())
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    plt.ylabel("number of users")
    plt.xlabel("interactions")
    plt.savefig(join(DATASET_DIR, "user_interactions_train.png"))

    # num unique user per author:
    train_unique_users_per_author = {k: len(v) for k, v in
                                    train.groupby(AUTHOR_FIELD)[USER_ID_FIELD].unique().to_dict().items()}
    valid_unique_users_per_author = {k: len(v) for k, v in
                                    valid.groupby(AUTHOR_FIELD)[USER_ID_FIELD].unique().to_dict().items()}
    test_unique_users_per_author = {k: len(v) for k, v in
                                    test.groupby(AUTHOR_FIELD)[USER_ID_FIELD].unique().to_dict().items()}

    # num unique author per user:
    train_unique_authors_per_user = train.groupby(USER_ID_FIELD)[AUTHOR_FIELD].nunique().to_dict()
    valid_unique_authors_per_user = valid.groupby(USER_ID_FIELD)[AUTHOR_FIELD].nunique().to_dict()
    test_unique_authors_per_user = test.groupby(USER_ID_FIELD)[AUTHOR_FIELD].nunique().to_dict()

    # combined dataset stats:
    temp = scipy.stats.describe(list({u: per_user['train'][u]+per_user['valid'][u]+per_user['test'][u] for u in per_user['train'].keys()}.values()))
    statfile.write(f"TOTAL: stats for user interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list({i: per_item['train'][i] + per_item['valid'][i] + per_item['test'][i] for i in per_item['train'].keys()}.values()))
    statfile.write(f"TOTAL: stats for item interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")

    temp = scipy.stats.describe(list({u: train_unique_authors_per_user[u]+valid_unique_authors_per_user[u]+test_unique_authors_per_user[u] for u in per_user['train'].keys()}.values()))
    statfile.write(
        f"TOTAL: avg num unique authors per user: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list({u: train_unique_users_per_author[u]+valid_unique_users_per_author[u]+test_unique_users_per_author[u] for u in per_user['train'].keys()}.values()))
    statfile.write(
        f"TOTAL: avg num unique users per author: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n\n\n")

    temp = scipy.stats.describe(list(per_user['train'].values()))
    statfile.write(f"TRAIN: stats for user interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(per_item['train'].values()))
    statfile.write(f"TRAIN: stats for item interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(f"TRAIN #interactions: {all_interactions_train} that is ratio {all_interactions_train/total_interactions}\n")
    statfile.write(f"TRAIN: num longlong tail users only in train: {len(per_user['train']) - len(per_user['test'])}"
                   f"with {user_grp_inter_cnt(train, set(per_user['train']) - set(per_user['test']), USER_ID_FIELD)} interactions.\n")
    temp = scipy.stats.describe(list(train_unique_authors_per_user.values()))
    statfile.write(
        f"TRAIN: avg num unique authors per user: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(train_unique_users_per_author.values()))
    statfile.write(
        f"TRAIN: avg num unique users per author: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(
        f"TRAIN: data sparsity 1-(#inter / #users*#items) = {1 - (all_interactions_train / (len(per_user['train']) * len(per_item['train'])))}\n")
    G, user_nodes = get_graph(train)
    statfile.write(f"TRAIN: is graph connected? {nx.is_connected(G)}\n")
    statfile.write(f"TRAIN: number of connected components: {nx.number_connected_components(G)}\n")
    statfile.write(f"TRAIN: density= {bipartite.density(G, user_nodes)}\n")
    statfile.write(f"TRAIN: sparsity= {1 - bipartite.density(G, user_nodes)}\n")
    for gr in user_groups:
        statfile.write(f"{gr}: {len(user_groups[gr])} users and "
                       f"{user_grp_inter_cnt(train, user_groups[gr], USER_ID_FIELD)}"
                       f" interactions\n")
    for gr in item_groups:
        statfile.write(f"{gr}: {len(item_groups[gr])} items and "
                       f"{user_grp_inter_cnt(train, item_groups[gr], ITEM_ID_FIELD)}"
                       f" interactions\n")
    statfile.write("\n")

    temp = scipy.stats.describe(list(per_user['valid'].values()))
    statfile.write(f"VALID: stats for user interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(per_item['valid'].values()))
    statfile.write(f"VALID: stats for item interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(f"VALID: #interactions: {all_interactions_valid} that is ratio {all_interactions_valid / total_interactions}\n")
    temp = scipy.stats.describe(list(valid_unique_authors_per_user.values()))
    statfile.write(
        f"VALID: avg num unique authors per user: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(valid_unique_users_per_author.values()))
    statfile.write(
        f"VALID: avg num unique users per author: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(f"VALID: num users in groups:\n")
    for gr in user_groups:
        statfile.write(f"{gr}: {len(valid_users.intersection(user_groups[gr]))} users and "
                       f"{user_grp_inter_cnt(valid, valid_users.intersection(user_groups[gr]), USER_ID_FIELD)}"
                       f" interactions\n")
    statfile.write(f"unseen: {len(valid_items - train_items)} items and "
                   f"{user_grp_inter_cnt(valid, valid_items - train_items, ITEM_ID_FIELD)}"
                   f" interactions\n")
    for gr in item_groups:
        statfile.write(f"{gr}: {len(valid_items.intersection(item_groups[gr]))} items and "
                       f"{user_grp_inter_cnt(valid, valid_items.intersection(item_groups[gr]), ITEM_ID_FIELD)}"
                       f" interactions\n")
    statfile.write("\n")

    temp = scipy.stats.describe(list(per_user['test'].values()))
    statfile.write(f"TEST: stats for user interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(per_item['test'].values()))
    statfile.write(f"TEST: stats for item interactions: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(f"TEST #interactions: {all_interactions_test} that is ratio {all_interactions_test / total_interactions}\n")
    temp = scipy.stats.describe(list(test_unique_authors_per_user.values()))
    statfile.write(
        f"TEST: avg num unique authors per user: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    temp = scipy.stats.describe(list(test_unique_users_per_author.values()))
    statfile.write(
        f"TEST: avg num unique users per author: {temp} - mean: {temp.mean.round(2)} - std: {math.sqrt(temp.variance).__round__(2)}\n")
    statfile.write(f"TEST: num users in groups:\n")
    for gr in user_groups:
        statfile.write(f"{gr}: {len(test_users.intersection(user_groups[gr]))} users and "
                       f"{user_grp_inter_cnt(test, test_users.intersection(user_groups[gr]), USER_ID_FIELD)}"
                       f" interactions\n")
    statfile.write(f"unseen: {len(test_items-train_items)} items and "
                   f"{user_grp_inter_cnt(test, test_items-train_items, ITEM_ID_FIELD)}"
                   f" interactions\n")
    for gr in item_groups:
        statfile.write(f"{gr}: {len(test_items.intersection(item_groups[gr]))} items and "
                       f"{user_grp_inter_cnt(test, test_items.intersection(item_groups[gr]), ITEM_ID_FIELD)}"
                       f" interactions\n")

### question: in k-fold cross-validation where we create the folds by users,
### all users exist in the train set. However, this is not the case for items,
### as we are not taking any especial considerations towards them.
