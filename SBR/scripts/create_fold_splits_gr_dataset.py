import csv
import os
import random
from os.path import join

import sklearn.model_selection
import numpy as np

rating_mapping = {
    '': 0,
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def filter_interactions(rating_th):
    interactions = []
    with open(join(RAW_DATASET_PATH, 'goodreads_crawled.interactions'), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for line in reader:
            if rating_th is not None:
                rating = rating_mapping[line[3]]
                if rating < rating_th:
                    continue
            interactions.append(line)
    return interactions, header


def get_user_interactions(inters):
    user_inters = {}
    for line in inters:
        user_id = line[1]
        if user_id not in user_inters:
            user_inters[user_id] = []
        user_inters[user_id].append(line)
    return user_inters


def get_long_tail_users(per_user_inters, inter_thr):
    long_tails = {}
    remaining = {}
    for user_id, user_inter in per_user_inters.items():
        if len(user_inter) <= inter_thr:
            long_tails[user_id] = user_inter
        else:
            remaining[user_id] = user_inter
    return long_tails, remaining


def create_folds(per_user_inters, n_folds, out_folder, inter_header):
    kfolder = sklearn.model_selection.KFold(n_splits=n_folds, shuffle=True, random_state=42)
    ret_folds = [[] for i in range(n_folds)]
    for user_id in per_user_inters.keys():
        user_inter = np.array(per_user_inters[user_id])
        for i, fold_inters in zip(range(n_folds), kfolder.split(user_inter)):
            ret_folds[i].extend(user_inter[fold_inters[1]].tolist())

    os.makedirs(join(out_folder, "folds"), exist_ok=True)
    for i in range(n_folds):
        with open(join(out_folder, "folds", f"fold_{i+1}.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(inter_header)
            writer.writerows(ret_folds[i])
    return ret_folds


def create_splits(fold_splits, directly_to_train_interactions, out_folder, inter_header, raw_dataset_path):
    # get the interactions going directly to train out of the loop as it is slow
    directly_to_train_rows = []
    dr_users = directly_to_train_interactions.keys()
    dr_items = []
    for user_id, user_inters in directly_to_train_interactions.items():
        directly_to_train_rows.extend(user_inters)
        # dr_items = dr_items.union(temp_items)
        dr_items.extend([line[2] for line in user_inters])
    dr_items = set(dr_items)

    # read user and item files:
    users_info = {}
    with open(join(raw_dataset_path, "goodreads_crawled.users"), 'r') as f:
        reader = csv.reader(f)
        user_file_header = next(reader)
        for line in reader:
            user_id = line[0]
            users_info[user_id] = line

    items_info = {}
    with open(join(raw_dataset_path, "goodreads_crawled.items"), 'r') as f:
        reader = csv.reader(f)
        item_file_header = next(reader)
        for line in reader:
            item_id = line[0]
            items_info[item_id] = line

    # create the splits:
    val_splits = set()
    for test_split_idx in range(len(fold_splits)):
        split_id = test_split_idx + 1
        os.makedirs(join(out_folder, f"split_{split_id}"), exist_ok=True)
        # here create train, valid, test interaction files, and (also) item and user files wrt the unique users/items.
        users = set()
        items = set()
        # write test_split:
        with open(join(out_folder, f"split_{split_id}", "test.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(inter_header)
            writer.writerows(fold_splits[test_split_idx])
        temp_users = set([line[1] for line in fold_splits[test_split_idx]])
        users = users.union(temp_users)
        temp_items = set([line[2] for line in fold_splits[test_split_idx]])
        items = items.union(temp_items)
        print(f"test idx: {test_split_idx}")

        # choose and write val split:
        val_split_idx = None
        while val_split_idx is None:
            temp = random.randint(0, len(fold_splits)-1)
            if temp not in val_splits and temp != test_split_idx:
                val_split_idx = temp
                val_splits.add(val_split_idx)
        with open(join(out_folder, f"split_{split_id}", "validation.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(inter_header)
            writer.writerows(fold_splits[val_split_idx])
        temp_users = set([line[1] for line in fold_splits[val_split_idx]])
        users = users.union(temp_users)
        temp_items = set([line[2] for line in fold_splits[val_split_idx]])
        items = items.union(temp_items)
        print(f"val idx: {val_split_idx}")

        # write the train split:
        with open(join(out_folder, f"split_{split_id}", "train.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(inter_header)
            for train_split_idx in range(len(fold_splits)):
                if train_split_idx not in [test_split_idx, val_split_idx]:
                    writer.writerows(fold_splits[train_split_idx])
                    temp_users = set([line[1] for line in fold_splits[train_split_idx]])
                    users = users.union(temp_users)
                    temp_items = set([line[2] for line in fold_splits[train_split_idx]])
                    items = items.union(temp_items)
                    print(f"train idx: {train_split_idx}")
            # direct ones:
            writer.writerows(directly_to_train_rows)
            users = users.union(dr_users)
            items = items.union(dr_items)

        print(f"users: {len(users)}")
        print(f"items: {len(items)}")

        # write current items and users files
        with open(join(out_folder, f"split_{split_id}", f"users.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(user_file_header)
            temp_rows = []
            for user_id in range(1, len(users_info)+1):
                if str(user_id) in users:
                    temp_rows.append(users_info[str(user_id)])
            writer.writerows(temp_rows)
        with open(join(out_folder, f"split_{split_id}", "items.csv"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(item_file_header)
            temp_rows = []
            for item_id in range(1, len(items_info) + 1):
                if str(item_id) in items:
                    temp_rows.append(items_info[str(item_id)])
            writer.writerows(temp_rows)


if __name__ == '__main__':
    random.seed(42)

    RAW_DATASET_PATH = "/home/ghazaleh/workspace/scrap_goodreads_CF_dataset/extracted_dataset/"

    # "None" if we want all signals, "0" if we want to include all interactions with rating, "3" if we want positive ratings:
    rating_threshold = None
    binary = True
    # We do 5 fold cross-validation:
    # 1: only from the users who have more or equal to 5 interactions such that they exist in every fold.
    # 2: we directly move the users with less than 5 interactions to the train set of every fold.
    # 3: 3-1-1
    # then see how the user interaction distribution seems in train/val/test sets.
    # we probably do not have enough interactions for cold users/items in the test/valid set.
    ## TODO if that did not work out, we decide to design the sets in the way that we want.
    num_folds = 5
    num_interactions_threshold = 4  # users with 4 or less interactions will directly go to train set

    OUTPUT_DATASET = f"data/GR_read_{num_folds}-folds/"
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    # filter interactions based on rating threshold
    interactions, header = filter_interactions(rating_threshold)

    # count interactions per user:
    per_user_interactions = get_user_interactions(interactions)

    # separate the interactions with less than threshold:
    # users with less than threshold interactions will be in the train split in all folds
    long_tail_user_interactions, to_be_split_interactions = \
        get_long_tail_users(per_user_interactions, num_interactions_threshold)
    print(f"long tail users: {len(long_tail_user_interactions)}")
    print(f"long tail interactions: {sum([len(v) for k, v in long_tail_user_interactions.items()])}")
    print(f"rest users: {len(to_be_split_interactions)}")
    print(f"rest interactions: {sum([len(v) for k, v in to_be_split_interactions.items()])}")
    # create n-folds with the remaining interactions:
    folds = create_folds(to_be_split_interactions, num_folds, OUTPUT_DATASET, header)

    # create n-splits with their train, valid, test sets.
    create_splits(folds, long_tail_user_interactions, OUTPUT_DATASET, header, RAW_DATASET_PATH)




### this is for binning data such that for example a query only exists in one fold
#     q_weights = {key: len(v) for key, v in q_inters.items()}
#     bins = binpacking.to_constant_bin_number(q_weights, n_folds)