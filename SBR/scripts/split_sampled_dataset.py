import csv
import os
import random
import shutil
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(per_user_interactions, ratios, longtail_trainonly_th):
    train = []
    test = []
    valid = []
    test_ratio = ratios[1]
    valid_ratio = ratios[2]/(1-test_ratio)  # e.g. [0.6, 0.2, 0.2] -> 0.2 = 0.25 * 0.8
    for user in per_user_interactions:
        if len(per_user_interactions[user]) <= longtail_trainonly_th:
            for line in per_user_interactions[user]:
                train.append(line)
        else:
            X_train, X_test = train_test_split(per_user_interactions[user],
                                                                test_size=test_ratio,
                                                                random_state=42)

            X_train, X_val = train_test_split(X_train,
                                                              test_size=valid_ratio,
                                                              random_state=42)
            for line in X_train:
                train.append(line)
            for line in X_test:
                test.append(line)
            for line in X_val:
                valid.append(line)

    return train, valid, test


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    DATASET_PATH = "TODO"
    INTERACTION_FILE = "goodreads_crawled.interactions"
    ITEM_FILE = "goodreads_crawled.items"
    USER_FILE = "goodreads_crawled.users"

    lt_threshold = 4
    ratios = [0.6, 0.2, 0.2]

    per_user_interactions = defaultdict(list)
    with open(join(DATASET_PATH, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        inter_header = next(reader)
        USER_ID_IDX_INTER = inter_header.index("user_id")
        ITEM_ID_IDX_INTER = inter_header.index("item_id")
        for line in reader:
            per_user_interactions[line[USER_ID_IDX_INTER]].append(line)

    train, valid, test = create_splits(per_user_interactions, ratios, lt_threshold)

    out_path = join(DATASET_PATH, f"ltth{lt_threshold}_ratios{'-'.join([str(r) for r in ratios])}")

    with open(join(out_path, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows(train)
    with open(join(out_path, "valid.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows(valid)
    with open(join(out_path, "test.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows(test)

    # copy user and item files
    shutil.copyfile(join(DATASET_PATH, ITEM_FILE), join(out_path, "items.csv"))
    shutil.copyfile(join(DATASET_PATH, USER_FILE), join(out_path, "users.csv"))

    allusers = set([line[USER_ID_IDX_INTER] for line in train])
    allusers = allusers.union(set([line[USER_ID_IDX_INTER] for line in test]))
    allusers = allusers.union(set([line[USER_ID_IDX_INTER] for line in valid]))
    print(f"num users: {len(allusers)}")

    allitems = set([line[ITEM_ID_IDX_INTER] for line in train])
    allitems = allitems.union(set([line[ITEM_ID_IDX_INTER] for line in test]))
    allitems = allitems.union(set([line[ITEM_ID_IDX_INTER] for line in valid]))
    print(f"num items: {len(allitems)}")
