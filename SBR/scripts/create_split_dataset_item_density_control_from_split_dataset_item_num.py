import argparse
import csv
import os
import random
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
    user_interaction_cnt = get_field_interaction_cnt(interactions, USER_ID_IDX_INTER)
    item_interaction_cnt = get_field_interaction_cnt(interactions, ITEM_ID_IDX_INTER)

    # starting randomly with a set of users:
    final_selected_items = set()
    cnt = 0
    while len(final_selected_items) < total_num_items:
        cnt += 1

        remaining_num_items = total_num_items - len(final_selected_items)
        si = round((starting_num_items * remaining_num_items) / total_num_items)
        if si == 0:  # corner case of only 1 user remaining...
            si = 1
        if si_objective == "random":
            chosen_h0_items = list(set(np.random.choice([i for i in item_interaction_cnt if i not in final_selected_items],
                                                       size=si,
                                                       replace=False)))
        else:
            if si_objective == "dense":
                si_degree = {item: item_interaction_cnt[item] for item in item_interaction_cnt if item not in final_selected_items}
            elif si_objective == "sparse":
                si_degree = {item: (1/item_interaction_cnt[item]) for item in item_interaction_cnt if item not in final_selected_items}
            else:
                raise ValueError("Not implemented")
            dem = sum(si_degree.values())
            chosen_h0_items = list(set(np.random.choice(list(si_degree.keys()),
                                                       size=si,
                                                       replace=False,
                                                       p=[p / dem for p in si_degree.values()])))
        if len(final_selected_items) + len(chosen_h0_items) > total_num_items:  # checking corner cases
            final_selected_items = final_selected_items.union(chosen_h0_items[:total_num_items - len(final_selected_items)])
            break
        final_selected_items = final_selected_items.union(chosen_h0_items)

        # expanding the chosen items wrt objective, selecting using propagation degree
        h0_users = set([l[USER_ID_IDX_INTER] for l in interactions if l[ITEM_ID_IDX_INTER] in chosen_h0_items])
        # we want to include the initial users, making sure connections
        if u_objective == "random":
            h0_users_degree = {u: 1 for u in h0_users}
        elif u_objective == "dense":
            h0_users_degree = {u: user_interaction_cnt[u] for u in h0_users}
        elif u_objective == "sparse":
            h0_users_degree = {u: (1/user_interaction_cnt[u]) for u in h0_users}
        else:
            raise ValueError("Not implemented")
        dem = sum(h0_users_degree.values())
        h0_users_probs = [p / dem for p in h0_users_degree.values()]
        h0_users_keys = list(h0_users_degree.keys())
        # we sample h1 users, and add them to the u0 users
        print(f"{cnt} users to choose from: {len(h0_users_keys)}")
        chosen_h0_users = list(np.random.choice(h0_users_keys, size=min(user_propagation_number, len(h0_users_keys)),
                                                replace=False, p=h0_users_probs))

        # expand and choose items up to item prop:
        h1_items = set([l[ITEM_ID_IDX_INTER] for l in interactions if l[USER_ID_IDX_INTER] in chosen_h0_users])
        h1_items = h1_items - final_selected_items
        # we want to include the initial users, making sure connections
        if i_objective == "random":
            h1_items_degree = {i: 1 for i in h1_items}
        elif i_objective == "dense":
            h1_items_degree = {i: item_interaction_cnt[i] for i in h1_items}
        elif i_objective == "sparse":
            h1_items_degree = {i: (1/item_interaction_cnt[i]) for i in h1_items}
        else:
            raise ValueError("Not implemented")
        dem = sum(h1_items_degree.values())
        h1_items_probs = [p / dem for p in h1_items_degree.values()]
        h1_items_keys = list(h1_items_degree.keys())
        print(f"{cnt} items to choose from: {len(h1_items_keys)}")
        chosen_h1_items = list(np.random.choice(h1_items_keys, size=min(item_propagation_number, len(h1_items_keys)),
                                                replace=False, p=h1_items_probs))
        if len(final_selected_items) + len(chosen_h1_items) > total_num_items:
            chosen_h1_items = chosen_h1_items[:total_num_items-len(final_selected_items)]
        final_selected_items = final_selected_items.union(chosen_h1_items)

    OUTPUT_DATASET = join(DATASET_PATH,
                          f'total-i-{total_num_items}_'
                          f'{cnt}_rounds_'
                          f'start-i-{starting_num_items}_{si_objective}_'
                          f'user-propag-{user_propagation_number}_{u_objective}_'
                          f'item-propag-{item_propagation_number}_{i_objective}')
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    total_users = set()
    for spf in INTERACTION_FILES:
        sp_rows = []
        with open(join(DATASET_PATH, spf), 'r') as f:
            reader = csv.reader(f)
            inter_header = next(reader)
            for line in reader:
                if line[ITEM_ID_IDX_INTER] in final_selected_items:
                    sp_rows.append(line)
                    total_users.add(line[USER_ID_IDX_INTER])

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
            if line[IDX] in total_users:
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
            if line[IDX] in final_selected_items:
                item_info.append(line)

    with open(join(OUTPUT_DATASET, ITEM_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_header)
        writer.writerows(item_info)


# run this on already splited data, for example the disjoint author case.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', '-d', type=str, default=None, help='path to dataset')
    parser.add_argument('--si', type=int, default=None, help='starting_num_items')
    parser.add_argument('--si_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--user_p', type=int, default=None, help='user_propagation_number')
    parser.add_argument('--u_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--item_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--i_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--total_items', '-i', type=int, default=None, help='total number of items to sample')
    args, _ = parser.parse_known_args()

    DATASET_PATH = args.dataset_path
    
    RATING_FIELD = "rating"
    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "item_id"
    INTERACTION_FILES = ["train.csv", "test.csv", "validation.csv"]
    ITEM_FILE = "items.csv"
    USER_FILE = "users.csv"

    starting_num_items = args.si
    si_objective = args.si_o
    user_propagation_number = args.user_p
    u_objective = args.u_o
    item_propagation_number = args.item_p
    i_objective = args.i_o
    total_num_items = args.total_items

    random.seed(42)
    np.random.seed(42)

    main()
