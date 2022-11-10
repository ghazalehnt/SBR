import json
import pickle
from collections import defaultdict
from os.path import join

import pandas as pd
import torch
import numpy as np


def main(outfile_item_user_distance, cf_checkpoint_file, cf_item_ids_file, train_file, valid_neg_file, test_neg_file):
    cf_item_reps = torch.load(cf_checkpoint_file, map_location=torch.device('cpu'))['model_state_dict']['item_embedding.weight']
    item_internal_ids = json.load(open(cf_item_ids_file, 'r'))
    train = pd.read_csv(train_file, dtype=str)

    user_eval_unlabeled_items = defaultdict(set)
    eval_negs = pd.read_csv(valid_neg_file, dtype=str)
    for user, item in zip(eval_negs['user_id'], eval_negs['item_id']):
        user_eval_unlabeled_items[user].add(item)
    eval_negs = pd.read_csv(test_neg_file, dtype=str)
    for user, item in zip(eval_negs['user_id'], eval_negs['item_id']):
        user_eval_unlabeled_items[user].add(item)

    item_item_similarity = {}
    num_users = len(set(train['user_id']))
    print(f"{num_users} users remained")
    for user in set(train['user_id']):
        user_items = list(set(train[train["user_id"] == user]["item_id"]))
        for unlabeled_item in user_eval_unlabeled_items[user]:
            for pos_item in user_items:
                if tuple(sorted([unlabeled_item, pos_item])) in item_item_similarity:
                    continue
                if sim_str == "dot":
                    s = np.dot(cf_item_reps[item_internal_ids[unlabeled_item]], cf_item_reps[item_internal_ids[pos_item]])
                elif sim_str == "cosine":
                    s = np.dot(cf_item_reps[item_internal_ids[unlabeled_item]],
                               cf_item_reps[item_internal_ids[pos_item]])\
                        / (np.linalg.norm(cf_item_reps[item_internal_ids[unlabeled_item]])
                           * np.linalg.norm(cf_item_reps[item_internal_ids[pos_item]]))
                item_item_similarity[tuple(sorted([unlabeled_item, pos_item]))] = s
        num_users -= 1
        if num_users % 100 == 0:
            print(f"{num_users} users remained")
    pickle.dump(item_item_similarity, open(outfile_item_user_distance, 'wb'))


if __name__ == "__main__":
    DATASET_DIR = ""
    VALID_NEG_FILE = "validation_neg_genres_2.csv"
    TEST_NEG_FILE = "test_neg_genres_2.csv"
    CF_CHECKPOINT_FILE = ""
    CF_ITEM_ID_FILE = ""
    # sim_str = "dot"
    sim_str = "cosine"

    main(join(DATASET_DIR, f"eval_item_ut_item_CF_sim_{sim_str}.pkl"),
         CF_CHECKPOINT_FILE,
         CF_ITEM_ID_FILE,
         join(DATASET_DIR, "train.csv"),
         join(DATASET_DIR, VALID_NEG_FILE),
         join(DATASET_DIR, TEST_NEG_FILE))
