import csv
from collections import defaultdict
from os.path import join

import networkx as nx
from networkx.algorithms import bipartite
import scipy.stats
import matplotlib.pyplot as plt


def get_histogram(vals):
    hist = {"1-4": 0, "5-10": 0, "11-20": 0, "21-30": 0, "31-40": 0, "41-50": 0, "51-60": 0, "61-70": 0, "71-80": 0,
              "81-90": 0, "91-100": 0, "101-200": 0, "201-500": 0, "501+": 0}
    for v in vals:
        if 1 <= v <= 4:
            hist["1-4"] += 1
        elif 5 <= v <= 10:
            hist["5-10"] += 1
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


def read_interactions(inter_file):
    ret = []
    with open(inter_file, 'r') as f:
        reader = csv.reader(f)
        h = next(reader)
        for line in reader:
            ret.append(line)
    return ret, h


def get_per_field_interaction_cnt(inters, field):
    ret = defaultdict(lambda : 0)
    for line in inters:
        field_id = line[field]
        ret[field_id] += 1
    return ret


def get_graph(inters):
    B = nx.Graph()
    user_nodes = set()
    item_nodes = set()
    edges = []
    for line in inters:
        user_id = f"u_{line[1]}"
        item_id = f"i_{line[2]}"
        user_nodes.add(user_id)
        item_nodes.add(item_id)
        edges.append((user_id, item_id))
    # Add nodes with the node attribute "bipartite"
    B.add_nodes_from(list(user_nodes), bipartite=0)
    B.add_nodes_from(list(item_nodes), bipartite=1)
    # Add edges only between nodes of opposite node sets
    B.add_edges_from(edges)
    return B, user_nodes


def get_user_groups(train_set, user_id_idx, thresholds=[5]):
    user_count = {}
    for line in train_set:
        if line[user_id_idx] not in user_count:
            user_count[line[user_id_idx]] = 1
        else:
            user_count[line[user_id_idx]] += 1
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


def user_grp_inter_cnt(split_set, users, user_id_idx):
    cnt = 0
    for line in split_set:
        if line[user_id_idx] in users:
            cnt += 1
    return cnt


if __name__ == '__main__':
    DATASET_DIR = ''
    USER_ID_FIELD = "reviewerID"
    ITEM_ID_FIELD = "asin"
    INTERACTION_FILE = "amazon_reviews_books.interactions"
    ITEM_FILE = "amazon_reviews_books.items"
    USER_FILE = "amazon_reviews_books.users"

    statfile = open(join(DATASET_DIR, f"stats.txt"), 'w')

    interactions, inter_header = read_interactions(join(DATASET_DIR, INTERACTION_FILE))
    # user_groups = get_user_groups(interactions, inter_header.index(USER_ID_FIELD), thresholds)

    per_user_inters = get_per_field_interaction_cnt(interactions, inter_header.index(USER_ID_FIELD))
    per_item_inters = get_per_field_interaction_cnt(interactions, inter_header.index(ITEM_ID_FIELD))

    print(f"stats for user interactions: {scipy.stats.describe(list(per_user_inters.values()))}")
    statfile.write(f"stats for user interactions: {scipy.stats.describe(list(per_user_inters.values()))}\n")
    print(f"stats for item interactions: {scipy.stats.describe(list(per_item_inters.values()))}")
    statfile.write(f"stats for item interactions: {scipy.stats.describe(list(per_item_inters.values()))}\n")

    sparsity_read = 1 - (len(interactions)/(len(per_user_inters.keys())*len(per_item_inters.keys())))
    print(f"read-matrix sparsity: {sparsity_read}")
    statfile.write(f"read-matrix sparsity: {sparsity_read}\n")

    hist = get_histogram(per_user_inters.values())
    x = list(hist.keys())
    y = list(hist.values())
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    plt.ylabel("number of users")
    plt.xlabel("interactions")
    plt.xticks(rotation=30)
    plt.savefig(join(DATASET_DIR, "user_interactions.png"))
    plt.clf()

    hist = get_histogram(per_item_inters.values())
    x = list(hist.keys())
    y = list(hist.values())
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    plt.ylabel("number of items")
    plt.xlabel("interactions")
    plt.xticks(rotation=30)
    plt.savefig(join(DATASET_DIR, "item_interactions.png"))
