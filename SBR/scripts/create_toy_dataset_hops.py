import csv
import os
import random
from os.path import join


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

    starting_num_users = 50
    num_lt_users = 20
    num_h1_items = 500
    total_num_users = 10000

    SPLIT_DATASET = f"{open('data/paths_vars/DATA_ROOT_PATH', 'r').read().strip()}/GR_read_5-folds/split_1/"

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
    all_users = set(chosen_users).union(chosen_long_tail_users)

    # grow chosen users by 2 hops
    h1_items = set([l[ITEM_ID_IDX] for l in train if l[USER_ID_IDX] in all_users])
    if num_h1_items < len(h1_items):
        chosen_h1_items = random.sample(list(h1_items), k=num_h1_items)
    else:
        chosen_h1_items = h1_items
    h1_users = set([l[USER_ID_IDX] for l in train if l[ITEM_ID_IDX] in chosen_h1_items])
    h1_users -= all_users
    if total_num_users-len(all_users) < len(h1_users):
        chosen_h1_users = random.sample(list(h1_users), k=total_num_users-len(all_users))
    else:
        chosen_h1_users = h1_users
    all_users = all_users.union(chosen_h1_users)

    OUTPUT_DATASET = f"{open('data/paths_vars/DATA_ROOT_PATH', 'r').read().strip()}/GR_read_5-folds/example_dataset_totalu{total_num_users}_su{starting_num_users}" \
                     f"_sltu{num_lt_users}_h1i{num_h1_items}/"
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
