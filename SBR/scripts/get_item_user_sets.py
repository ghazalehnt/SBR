import json
import pickle
from collections import defaultdict
from os.path import join

import pandas as pd

def jaccard_index(X, Y):
    X = set(X)
    return len(X.intersection(Y))/len(X.union(Y))


def main(interaction_file, user_idx_field, item_idx_field, rating_field, pos_thr, outfile_item_user_set,
         outfile_item_item_distance):
    df = pd.read_csv(interaction_file, usecols=[user_idx_field, item_idx_field, rating_field], dtype=str)
    df[rating_field] = df[rating_field].astype(float).astype(int)
    df = df[df[rating_field] >= pos_thr]
    item_user_set = defaultdict(list)
    for user, item in zip(df[user_idx_field], df[item_idx_field]):
        item_user_set[item].append(user)
    pickle.dump(item_user_set, open(outfile_item_user_set, 'wb'))

    # let's also calculate the item-item jaccard index
    distances = defaultdict()
    items = sorted(list(item_user_set.keys()))
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            i1 = items[i]
            i2 = items[j]
            distance = jaccard_index(item_user_set[i1], item_user_set[i2])
            distances[(i1, i2)] = distance
    pickle.dump(item_user_set, open(outfile_item_item_distance, 'wb'))


if __name__ == "__main__":
    DATASET_DIR = ""
    INTER_FILE = "train.csv"
    USER_IDX_FIELD = "user_id"
    ITEM_IDX_FIELD = "item_id"
    RATING_FIELD = "rating"
    POS_TH = 3

    main(join(DATASET_DIR, INTER_FILE), USER_IDX_FIELD, ITEM_IDX_FIELD, RATING_FIELD, POS_TH,
         join(DATASET_DIR, "item_user_set.pkl"), join(DATASET_DIR, "item_item_jaccard_index_distance.pkl"))