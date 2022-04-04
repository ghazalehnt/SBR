import csv
import os
import random
from os.path import join

import sklearn.model_selection
import numpy as np


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

## TODO this toy example is not good, users are random they may have no ineraction with each other!!!!

if __name__ == "__main__":
    random.seed(42)
    SPLIT_DATASET = "data/GR_read_5-folds/split_1/"
    OUTPUT_DATASET = "data/GR_read_5-folds/toy_dataset/"
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    train, valid, test, inter_header = read_interactions(SPLIT_DATASET)
    print(inter_header)
    USER_ID_IDX = inter_header.index("user_id")
    ITEM_ID_IDX = inter_header.index("item_id")

    train_users = set([l[USER_ID_IDX] for l in train])
    test_users = set([l[USER_ID_IDX] for l in test])
    long_tail_users = list(train_users - test_users)
    shared_users = list(test_users)

    chosen_long_tail_users = random.choices(long_tail_users, k=20)
    chosen_users = random.choices(shared_users, k=100)

    all_items_original = set()
    new_user_ids = [-1]
    new_item_ids = [-1]
    with open(join(OUTPUT_DATASET, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in train:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in chosen_users or user_id in chosen_long_tail_users:
                all_items_original.add(item_id)

                if user_id not in new_user_ids:
                    new_user_ids.append(user_id)
                new_user_id = new_user_ids.index(user_id)
                line[USER_ID_IDX] = new_user_id

                if item_id not in new_item_ids:
                    new_item_ids.append(item_id)
                new_item_id = new_item_ids.index(item_id)
                line[ITEM_ID_IDX] = new_item_id

                temp.append(line)

        writer.writerows(temp)

    with open(join(OUTPUT_DATASET, "validation.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in valid:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in chosen_users:
                all_items_original.add(item_id)

                if user_id not in new_user_ids:
                    new_user_ids.append(user_id)
                new_user_id = new_user_ids.index(user_id)
                line[USER_ID_IDX] = new_user_id

                if item_id not in new_item_ids:
                    new_item_ids.append(item_id)
                new_item_id = new_item_ids.index(item_id)
                line[ITEM_ID_IDX] = new_item_id

                temp.append(line)
        writer.writerows(temp)

    with open(join(OUTPUT_DATASET, "test.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in test:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in chosen_users:
                all_items_original.add(item_id)

                if user_id not in new_user_ids:
                    new_user_ids.append(user_id)
                new_user_id = new_user_ids.index(user_id)
                line[USER_ID_IDX] = new_user_id

                if item_id not in new_item_ids:
                    new_item_ids.append(item_id)
                new_item_id = new_item_ids.index(item_id)
                line[ITEM_ID_IDX] = new_item_id

                temp.append(line)
        writer.writerows(temp)

    # read/write user and item files:
    user_info = []
    with open(join(SPLIT_DATASET, f"users.csv"), 'r') as f:
        reader = csv.reader(f)
        user_header = next(reader)
        IDX = user_header.index("user_id")
        for line in reader:
            if line[IDX] in chosen_users or line[IDX] in chosen_long_tail_users:
                line[IDX] = new_user_ids.index(line[IDX])
                user_info.append(line)
    item_info = []
    with open(join(SPLIT_DATASET, f"items.csv"), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        IDX = item_header.index("item_id")
        for line in reader:
            if line[IDX] in all_items_original:
                line[IDX] = new_item_ids.index(line[IDX])
                item_info.append(line)

    with open(join(OUTPUT_DATASET, f"users.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(user_header)
        writer.writerows(user_info)

    with open(join(OUTPUT_DATASET, "items.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_header)
        writer.writerows(item_info)
