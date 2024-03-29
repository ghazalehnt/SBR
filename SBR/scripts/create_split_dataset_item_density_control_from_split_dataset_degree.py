import argparse
import csv
import os
import random
import subprocess
from collections import defaultdict
from os.path import join
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)


def get_field_interaction_cnt(inters, field):
    ret = defaultdict(lambda: 0)
    for line in inters:
        field_id = line[field]
        ret[field_id] += 1
    return ret


def get_user_degree(inters, item_interaction_cnt, user_field, item_field):
    ret = defaultdict(lambda: 0)
    for line in inters:
        user_id = line[user_field]
        item_id = line[item_field]
        ret[user_id] += item_interaction_cnt[item_id]
    return ret


def get_interactions():
    interactions = []
    for spf in INTERACTION_FILES:
        with open(join(DATASET_PATH, spf), 'r') as f:
            reader = csv.reader(f)
            inter_header = next(reader)
            for line in reader:
                interactions.append(line)
        print(inter_header)
        print(len(interactions))
    return interactions, inter_header


def main():
    interactions, inter_header = get_interactions()
    USER_ID_IDX_INTER = inter_header.index(USER_ID_FIELD)
    ITEM_ID_IDX_INTER = inter_header.index(ITEM_ID_FIELD)
    item_interaction_cnt = get_field_interaction_cnt(interactions, ITEM_ID_IDX_INTER)
    user_item_degree = get_user_degree(interactions, item_interaction_cnt, USER_ID_FIELD)

    # starting randomly with a set of users:
    final_selected_users = set()
    cnt = 0
    while len(final_selected_users) < total_num_users:
        cnt += 1

        if si_objective == "random":
            chosen_h0_items = list(set(np.random.choice(list(item_interaction_cnt),
                                                       size=starting_num_items,
                                                       replace=False)))
        else:
            if si_objective == "dense":
                si_degree = {item: item_interaction_cnt[item] for item in item_interaction_cnt}
            elif si_objective == "sparse":
                si_degree = {item: (1/item_interaction_cnt[item]) for item in item_interaction_cnt}
            else:
                raise ValueError("Not implemented")
            dem = sum(si_degree.values())
            chosen_h0_items = list(set(np.random.choice(list(si_degree.keys()),
                                                       size=starting_num_items,
                                                       replace=False,
                                                       p=[p / dem for p in si_degree.values()])))

        # expanding the chosen items wrt objective, selecting using propagation degree
        h0_users = set([l[USER_ID_IDX_INTER] for l in interactions if l[ITEM_ID_IDX_INTER] in chosen_h0_items])
        # we want to include the initial users, making sure connections
        h0_users = h0_users - final_selected_users  # select new users
        if u_objective == "random":
            h0_users_degree = {u: 1 for u in h0_users}
        elif u_objective == "dense":
            h0_users_degree = {u: user_item_degree[u] for u in h0_users}
        elif u_objective == "sparse":
            h0_users_degree = {u: (1/user_item_degree[u]) for u in h0_users}
        else:
            raise ValueError("Not implemented")
        dem = sum(h0_users_degree.values())
        h0_users_probs = [p / dem for p in h0_users_degree.values()]
        h0_users_keys = list(h0_users_degree.keys())
        # we sample h1 users, and add them to the u0 users
        print(f"{cnt} users to choose from: {len(h0_users_keys)}")
        chosen_h0_users = list(np.random.choice(h0_users_keys, size=min(user_propagation_number, len(h0_users_keys)),
                                                replace=False, p=h0_users_probs))

        if len(final_selected_users) + len(chosen_h0_users) > total_num_users:
            chosen_h0_users = chosen_h0_users[:total_num_users-len(final_selected_users)]
        final_selected_users = final_selected_users.union(chosen_h0_users)

    OUTPUT_DATASET = join(DATASET_PATH,
                          f'total-u-{total_num_users}_'
                          f'{cnt}_rounds_'
                          f'start-i-{starting_num_items}_{si_objective}_'
                          f'user-propag-{user_propagation_number}_{u_objective}')
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    total_items = set()

    for spf in INTERACTION_FILES:
        sp_rows = []
        with open(join(DATASET_PATH, spf), 'r') as f:
            reader = csv.reader(f)
            inter_header = next(reader)
            for line in reader:
                if line[USER_ID_IDX_INTER] in final_selected_users:
                    sp_rows.append(line)
                    total_items.add(line[ITEM_ID_IDX_INTER])

        with open(join(OUTPUT_DATASET, spf), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(inter_header)
            writer.writerows(sp_rows)

    # read/write user and item files:
    user_info = []
    with open(join(DATASET_PATH, USER_FILE), 'r') as f:
        reader = csv.reader(f)
        user_header = next(reader)
        IDX = user_header.index(USER_ID_FIELD)
        for line in reader:
            if line[IDX] in final_selected_users:
                user_info.append(line)

    with open(join(OUTPUT_DATASET, USER_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(user_header)
        writer.writerows(user_info)

    item_info = []
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        IDX = item_header.index(ITEM_ID_FIELD)
        for line in reader:
            if line[IDX] in total_items:
                item_info.append(line)

    with open(join(OUTPUT_DATASET, ITEM_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_header)
        writer.writerows(item_info)

    print(OUTPUT_DATASET)
    auth = "authors"
    cmd = f'python /GW/PSR/work/ghazaleh/SBR/SBR/scripts/get_stats_train_dev_test_splits.py -d {OUTPUT_DATASET} --af {auth}'
    subprocess.call(cmd, shell=True)


# run this on already splited data, for example the disjoint author case.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default=None, help='path to dataset')
    parser.add_argument('--si', type=int, default=None, help='starting_num_items')
    parser.add_argument('--si_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--user_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--u_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--total_users', '-u', type=int, default=None, help='total number of users to sample')
    args, _ = parser.parse_known_args()

    DATASET_PATH = args.dataset_path
    
    RATING_FIELD = "rating"
    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "item_id"
    INTERACTION_FILES = ["train.csv", "test.csv", "validation.csv"]
    ITEM_FILE = "items.csv"
    USER_FILE = "users.csv"

    starting_num_items = args.si
    user_propagation_number = args.user_p
    total_num_users = args.total_users

    u_objective = args.u_o
    si_objective = args.si_o

    random.seed(42)
    np.random.seed(42)

    main()
