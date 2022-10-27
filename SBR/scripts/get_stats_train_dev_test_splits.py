import csv
from os.path import join

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
        ret[split] = []
        with open(join(dataset_path, sp_file), 'r') as f:
            reader = csv.reader(f)
            h = next(reader)
            for line in reader:
                ret[split].append(line)
    return ret["train"], ret["validation"], ret["test"], h


def get_per_user_interaction_cnt(inters):
    ret = {}
    for line in inters:
        user_id = line[1]
        if user_id not in ret:
            ret[user_id] = 0
        ret[user_id] += 1
    return ret


def get_per_item_interaction_cnt(inters):
    ret = {}
    for line in inters:
        item_id = line[2]
        if item_id not in ret:
            ret[item_id] = 0
        ret[item_id] += 1
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
    # DATASET_DIR = f"{open('data/paths_vars/DATA_ROOT_PATH', 'r').read().strip()}/GR_rating3_5-folds/example_dataset_totalu10000_su100_sltu100_h1i500_dense"
    USER_ID_FIELD = "user_id"
    thresholds = [3, 6, 10, 100]
    statfile = open(join(DATASET_DIR, f"stats_{'-'.join([str(thr) for thr in thresholds])}.txt"), 'w')

    train, valid, test, header = read_interactions(DATASET_DIR)
    user_groups = get_user_groups(train, header.index(USER_ID_FIELD), thresholds)
    test_users = set([line[header.index(USER_ID_FIELD)] for line in test])
    valid_users = set([line[header.index(USER_ID_FIELD)] for line in valid])

    per_user = {"train": get_per_user_interaction_cnt(train),
                "valid": get_per_user_interaction_cnt(valid),
                "test": get_per_user_interaction_cnt(test)}

    per_item = {"train": get_per_item_interaction_cnt(train),
                "valid": get_per_item_interaction_cnt(valid),
                "test": get_per_item_interaction_cnt(test)}

    all_interactions = sum(per_user['train'].values())

    h_user = get_histogram(per_user['train'].values())
    x = list(h_user.keys())
    y = list(h_user.values())
    plt.bar(x, y)
    for i in range(len(x)):
        plt.text(i, y[i], y[i])
    plt.ylabel("number of users")
    plt.xlabel("interactions")
    plt.savefig(join(DATASET_DIR, "user_interactions_train.png"))

    statfile.write(f"TRAIN: stats for user interactions: {scipy.stats.describe(list(per_user['train'].values()))}\n")
    statfile.write(f"TRAIN: stats for item interactions: {scipy.stats.describe(list(per_item['train'].values()))}\n")
    statfile.write(f"TRAIN: num longlong tail users only in train: {len(per_user['train']) - len(per_user['test'])}\n")
    statfile.write(
        f"TRAIN: data sparsity 1-(#inter / #users*#items) = {1 - (all_interactions / (len(per_user['train']) * len(per_item['train'])))}\n")
    G, user_nodes = get_graph(train)
    statfile.write(f"TRAIN: is graph connected? {nx.is_connected(G)}\n")
    statfile.write(f"TRAIN: number of connected components: {nx.number_connected_components(G)}\n")
    statfile.write(f"TRAIN: density= {bipartite.density(G, user_nodes)}\n")
    statfile.write(f"TRAIN: sparsity= {1 - bipartite.density(G, user_nodes)}\n\n")

    statfile.write(f"VALID: stats for user interactions: {scipy.stats.describe(list(per_user['valid'].values()))}\n")
    statfile.write(f"VALID: stats for item interactions: {scipy.stats.describe(list(per_item['valid'].values()))}\n")
    statfile.write(f"VALID: num users in groups:\n")
    for gr in user_groups:
        statfile.write(f"{gr}: {len(valid_users.intersection(user_groups[gr]))} users and "
                       f"{user_grp_inter_cnt(valid, valid_users.intersection(user_groups[gr]), header.index(USER_ID_FIELD))}"
                       f" interactions\n")

    statfile.write(f"TEST: stats for user interactions: {scipy.stats.describe(list(per_user['test'].values()))}\n")
    statfile.write(f"TEST: stats for item interactions: {scipy.stats.describe(list(per_item['test'].values()))}\n")
    statfile.write(f"VALID: num users in groups:\n")
    for gr in user_groups:
        statfile.write(f"{gr}: {len(test_users.intersection(user_groups[gr]))} users and "
                       f"{user_grp_inter_cnt(test, test_users.intersection(user_groups[gr]), header.index(USER_ID_FIELD))}"
                       f" interactions\n")

### question: in k-fold cross-validation where we create the folds by users,
### all users exist in the train set. However, this is not the case for items,
### as we are not taking any especial considerations towards them.