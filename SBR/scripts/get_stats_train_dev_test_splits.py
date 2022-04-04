import csv
from os.path import join

import scipy.stats
# import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    DATASET_DIR = "data/GR_read_5-folds/split_1/"
    statfile = open(join(DATASET_DIR, "stats.txt"), 'w')

    train, valid, test, header = read_interactions(DATASET_DIR)

    per_user = {"train": get_per_user_interaction_cnt(train),
                "valid": get_per_user_interaction_cnt(valid),
                "test": get_per_user_interaction_cnt(test)}

    per_item = {"train": get_per_item_interaction_cnt(train),
                "valid": get_per_item_interaction_cnt(valid),
                "test": get_per_item_interaction_cnt(test)}

    statfile.write(f"TRAIN: stats for user interactions: {scipy.stats.describe(list(per_user['train'].values()))}\n")
    statfile.write(f"TRAIN: stats for item interactions: {scipy.stats.describe(list(per_item['train'].values()))}\n")

    statfile.write(f"VALID: stats for user interactions: {scipy.stats.describe(list(per_user['valid'].values()))}\n")
    statfile.write(f"VALID: stats for item interactions: {scipy.stats.describe(list(per_item['valid'].values()))}\n")

    statfile.write(f"TEST: stats for user interactions: {scipy.stats.describe(list(per_user['test'].values()))}\n")
    statfile.write(f"TEST: stats for item interactions: {scipy.stats.describe(list(per_item['test'].values()))}\n")

### question: in k-fold cross-validation where we create the folds by users,
### all users exist in the train set. However, this is not the case for items,
### as we are not taking any especial considerations towards them.