import pickle
from collections import defaultdict
from os.path import join

import pandas as pd


def jaccard_index(item_user_set, item1, item2):
    X = set(item_user_set[item1])
    Y = item_user_set[item2]
    d = len(X.intersection(Y))/len(X.union(Y))
    return d


def main(outfile_item_user_distance, item_user_set_file, train_file, valid_neg_file, test_neg_file):
    item_user_set = pickle.load(open(item_user_set_file, 'rb'))
    train = pd.read_csv(train_file, dtype=str)

    user_eval_unlabeled_items = defaultdict(set)
    eval_negs = pd.read_csv(valid_neg_file, dtype=str)
    for user, item in zip(eval_negs['user_id'], eval_negs['item_id']):
        user_eval_unlabeled_items[user].add(item)
    eval_negs = pd.read_csv(test_neg_file, dtype=str)
    for user, item in zip(eval_negs['user_id'], eval_negs['item_id']):
        user_eval_unlabeled_items[user].add(item)

    item_user_distance = defaultdict(dict)
    for user in set(train['user_id']):
        user_items = list(set(train[train["user_id"] == user]["item_id"]))

        for unlabeled_item in user_eval_unlabeled_items[user]:
            dist = [jaccard_index(item_user_set, unlabeled_item, pos_item) for pos_item in user_items]
            avgdist = sum(dist) / len(dist)
            item_user_distance[user][unlabeled_item] = avgdist
    pickle.dump(item_user_distance, open(outfile_item_user_distance, 'wb'))


if __name__ == "__main__":
    DATASET_DIR = ""
    VALID_NEG_FILE = "validation_neg_genres_2.csv"
    TEST_NEG_FILE = "test_neg_genres_2.csv"
    ITEM_USER_PATH = ""

    main(join(DATASET_DIR, "eval_user_item_jaccard_index.pkl"),
         ITEM_USER_PATH,
         join(DATASET_DIR, "train.csv"),
         join(DATASET_DIR, VALID_NEG_FILE),
         join(DATASET_DIR, TEST_NEG_FILE))