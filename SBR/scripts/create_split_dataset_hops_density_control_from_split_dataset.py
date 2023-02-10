import argparse
import csv
import os
import random
from os.path import join
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)


def get_per_user_interaction_cnt(inters, USER_ID_IDX):
    ret = {}
    for line in inters:
        user_id = line[USER_ID_IDX]
        if user_id not in ret:
            ret[user_id] = 0
        ret[user_id] += 1
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
    user_interaction_cnt = get_per_user_interaction_cnt(interactions, USER_ID_IDX_INTER)

    # starting randomly with a set of users:
    final_selected_users = set(np.random.choice(list(user_interaction_cnt.keys()), size=starting_num_users, replace=False))
    hop = 0
    while len(final_selected_users) < total_num_users:
        hop += 1
        # then selecting their items wrt the objective and choosing a number of them wrt propagation degree
        h0_items = set([l[ITEM_ID_IDX_INTER] for l in interactions if l[USER_ID_IDX_INTER] in final_selected_users])
        if objective == "random":
            h0_items_degree = {item: 1 for item in h0_items}
        else:
            h0_items_degree = {item: 0 for item in h0_items}
            for l in interactions:
                if l[ITEM_ID_IDX_INTER] in h0_items:
                    h0_items_degree[l[ITEM_ID_IDX_INTER]] += 1

            if objective == "dense":
                h0_items_degree = {k: v for k, v in h0_items_degree.items()}
            elif objective == "sparse":
                h0_items_degree = {k: (1 / v) for k, v in h0_items_degree.items()}
            else:
                raise ValueError("Not implemented")
        dem = sum(h0_items_degree.values())
        h0_items_probs = [p / dem for p in h0_items_degree.values()]
        h0_items_keys = list(h0_items_degree.keys())
        print(f"{hop} items to choose from: {len(h0_items_keys)}")
        chosen_h0_items = list(np.random.choice(h0_items_keys, size=item_propagation_number, replace=False, p=h0_items_probs))

        # expanding the chosen items wrt objective, selecting using propagation degree
        h1_users = set([l[USER_ID_IDX_INTER] for l in interactions if l[ITEM_ID_IDX_INTER] in chosen_h0_items])
        # we want to include the initial users, making sure connections
        h1_users = h1_users - final_selected_users
        if objective == "random":
            h1_users_degree = {u: 1 for u in h1_users}
        else:
            h1_users_degree = {u: 0 for u in h1_users}
            for l in interactions:
                if l[USER_ID_IDX_INTER] in h1_users:
                    h1_users_degree[l[USER_ID_IDX_INTER]] += 1
            if objective == "dense":
                h1_users_degree = {k: v for k, v in h1_users_degree.items()}
            elif objective == "sparse":
                h1_users_degree = {k: (1 / v) for k, v in h1_users_degree.items()}
            else:
                raise ValueError("Not implemented")
        dem = sum(h1_users_degree.values())
        h1_users_probs = [p / dem for p in h1_users_degree.values()]
        h1_users_keys = list(h1_users_degree.keys())
        # we sample h1 users, and add them to the u0 users
        print(f"{hop} users to choose from: {len(h1_users_keys)}")
        chosen_h1_users = list(np.random.choice(h1_users_keys, size=user_propagation_number, replace=False, p=h1_users_probs))

        if len(final_selected_users) + len(chosen_h1_users) > total_num_users:
            chosen_h1_users = chosen_h1_users[:total_num_users-len(final_selected_users)]
        final_selected_users = final_selected_users.union(chosen_h1_users)

    OUTPUT_DATASET = join(DATASET_PATH,
                          f'total-u-{total_num_users}_'
                          f'start-u-{starting_num_users}_'
                          f'item-propag-{item_propagation_number}_'
                          f'user-propag-{user_propagation_number}_'
                          f'{objective}')
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


# run this on already splited data, for example the disjoint author case.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default=None, help='path to dataset')
    parser.add_argument('--su', type=int, default=None, help='starting_num_users')
    parser.add_argument('--item_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--user_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--total_users', '-u', type=int, default=None, help='total number of users to sample')
    parser.add_argument('--objective', '-o', type=str, default=None, help='random/sparse/dense')
    args, _ = parser.parse_known_args()

    random.seed(42)
    np.random.seed(42)

    DATASET_PATH = args.dataset_path

    RATING_FIELD = "rating"
    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "item_id"
    INTERACTION_FILES = ["train.csv", "test.csv", "validation.csv"]
    ITEM_FILE = "items.csv"
    USER_FILE = "users.csv"

    starting_num_users = args.su
    item_propagation_number = args.item_p
    user_propagation_number = args.user_p
    total_num_users = args.total_users
    objective = args.objective

    main()
