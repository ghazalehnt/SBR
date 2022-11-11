import pickle
from collections import defaultdict
from os.path import join

import pandas as pd


def main(outfile_item_user_sim_file, train_file, dataset_dir, neg_files, item_sim_file, oldmin, oldmax):
    item_item_similarity = pickle.load(open(item_sim_file, 'rb'))
    train = pd.read_csv(train_file, dtype=str)

    user_eval_unlabeled_items = defaultdict(set)
    for neg_file in neg_files:
        eval_negs = pd.read_csv(join(dataset_dir, neg_file), dtype=str)
        for user, item in zip(eval_negs['user_id'], eval_negs['item_id']):
            user_eval_unlabeled_items[user].add(item)

    item_user_similarity = defaultdict(dict)
    for user in set(train['user_id']):
        user_items = list(set(train[train["user_id"] == user]["item_id"]))
        for unlabeled_item in user_eval_unlabeled_items[user]:
            sims = []
            for pos_item in user_items:
                s = item_item_similarity[tuple(sorted([unlabeled_item, pos_item]))]
                s = (s - oldmin) / (oldmax - oldmin)  # s = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
                # as oldmin and oldmax are estimates, we make sure that the s is between 0 and 1:
                s = max(0, s)
                s = min(1, s)
                sims.append(s)
            avg_sim = sum(sims) / len(sims)
            item_user_similarity[user][unlabeled_item] = avg_sim
    pickle.dump(item_user_similarity, open(outfile_item_user_sim_file, 'wb'))


if __name__ == "__main__":
    DATASET_DIR = ""
    neg_files = ["validation_neg_genres_100.csv", "test_neg_genres_100.csv",
                 "validation_neg_random_100.csv", "test_neg_random_100.csv"]
    ITEM_ITEM_SIM_FILE = ""
    sim_str = "dot"
    minold = 0
    maxold = 71

    main(join(DATASET_DIR, f"eval_user_item_CF_sim_scaled_{minold}-{maxold}.pkl"),
         join(DATASET_DIR, "train.csv"),
         DATASET_DIR,
         neg_files,
         join(DATASET_DIR, ITEM_ITEM_SIM_FILE),
         minold,
         maxold)
