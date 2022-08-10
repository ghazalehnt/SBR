import csv
import os
import random
from os.path import join
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


## Here we randomly select N users, then expand them by max 2 hops.
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    dataset_slice = "GR_rating3_5-folds"
    original_split = "split_1"
    starting_num_users = 50
    num_lt_users = 20
    num_h0_items = 500
    total_num_users = 10000
    # objective = 'dense'
    # objective = 'sparse'
    objective = 'random'

    SPLIT_DATASET = f"{open('data/paths_vars/DATA_ROOT_PATH', 'r').read().strip()}/{dataset_slice}/{original_split}"

    train, valid, test, inter_header = read_interactions(SPLIT_DATASET)
    print(inter_header)
    USER_ID_IDX = inter_header.index("user_id")
    ITEM_ID_IDX = inter_header.index("item_id")

    train_users = set([l[USER_ID_IDX] for l in train])
    test_users = set([l[USER_ID_IDX] for l in test])
    long_tail_users = list(train_users - test_users)
    shared_users = list(test_users)

    chosen_long_tail_users = random.sample(long_tail_users, k=num_lt_users)
    chosen_users = random.sample(shared_users, k=starting_num_users)
    h0_users = set(chosen_users).union(chosen_long_tail_users)

    # grow chosen users by 2 hops
    h0_items = set([l[ITEM_ID_IDX] for l in train if l[USER_ID_IDX] in h0_users])
    if objective == "random":
        h0_items_degree = {item: 1 for item in h0_items}
    else:
        h0_items_degree = {item: 0 for item in h0_items}
        for l in train:
            if l[ITEM_ID_IDX] in h0_items:
                h0_items_degree[l[ITEM_ID_IDX]] += 1
        if objective == "dense":
            h0_items_degree = {k: v for k, v in h0_items_degree.items()}
        elif objective == "sparse":
            h0_items_degree = {k: (1/v) for k, v in h0_items_degree.items()}
        else:
            raise ValueError("Not implemented")
    dem = sum(h0_items_degree.values())
    h0_items_probs = [p/dem for p in h0_items_degree.values()]
    h0_items_keys = list(h0_items_degree.keys())
    print(len(h0_items_keys))
    chosen_h0_items = list(np.random.choice(h0_items_keys, size=num_h0_items, replace=False, p=h0_items_probs))

    h1_users = set([l[USER_ID_IDX] for l in train if l[ITEM_ID_IDX] in chosen_h0_items])
    # we want to include the initial users, making sure connections
    h1_users = h1_users - h0_users

    if objective == "random":
        h1_users_degree = {u: 1 for u in h1_users}
    else:
        h1_users_degree = {u: 0 for u in h1_users}
        for l in train:
            if l[USER_ID_IDX] in h1_users:
                h1_users_degree[l[USER_ID_IDX]] += 1
        if objective == "dense":
            h1_users_degree = {k: v for k, v in h1_users_degree.items()}
        elif objective == "sparse":
            h1_users_degree = {k: (1/v) for k, v in h1_users_degree.items()}
        else:
            raise ValueError("Not implemented")
    dem = sum(h1_users_degree.values())
    h1_users_probs = [p/dem for p in h1_users_degree.values()]
    h1_users_keys = list(h1_users_degree.keys())

    # we sample h1 users, and add them to the u0 users
    print(len(h1_users_keys))
    chosen_h1_users = list(np.random.choice(h1_users_keys, size=total_num_users-len(h0_users), replace=False, p=h1_users_probs))
    all_users = h0_users.union(chosen_h1_users)

    OUTPUT_DATASET = f"{open('data/paths_vars/DATA_ROOT_PATH', 'r').read().strip()}/{dataset_slice}/example_dataset_totalu{total_num_users}_su{starting_num_users}" \
                     f"_sltu{num_lt_users}_h1i{num_h0_items}_{objective}/"
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    total_items = set()
    total_users = set(all_users)
    with open(join(OUTPUT_DATASET, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in train:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in all_users:
                total_items.add(item_id)
                temp.append(line)
        writer.writerows(temp)

    with open(join(OUTPUT_DATASET, "validation.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in valid:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in all_users:
                total_items.add(item_id)
                temp.append(line)
        writer.writerows(temp)

    with open(join(OUTPUT_DATASET, "test.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in test:
            user_id = line[USER_ID_IDX]
            item_id = line[ITEM_ID_IDX]
            if user_id in all_users:
                total_items.add(item_id)
                temp.append(line)
        writer.writerows(temp)

    # read/write user and item files:
    user_info = []
    with open(join(SPLIT_DATASET, f"users.csv"), 'r') as f:
        reader = csv.reader(f)
        user_header = next(reader)
        IDX = user_header.index("user_id")
        for line in reader:
            if line[IDX] in total_users:
                user_info.append(line)
    item_info = []
    with open(join(SPLIT_DATASET, f"items.csv"), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        IDX = item_header.index("item_id")
        for line in reader:
            if line[IDX] in total_items:
                item_info.append(line)

    with open(join(OUTPUT_DATASET, f"users.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(user_header)
        writer.writerows(user_info)

    with open(join(OUTPUT_DATASET, "items.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_header)
        writer.writerows(item_info)
