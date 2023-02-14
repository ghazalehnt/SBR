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
    final_selected_users = set()
    cnt = 0
    while len(final_selected_users) < total_num_users:
        cnt += 1

        remaining_num_user = total_num_users - len(final_selected_users)
        su = round((starting_num_users * remaining_num_user) / total_num_users)
        if su == 0:  # corner case of only 1 user remaining...
            su = 1

        if su_objective == "random":
            su_degree = {user: 1 for user in user_interaction_cnt if user not in final_selected_users}
        elif su_objective == "dense":
            su_degree = {user: user_interaction_cnt[user] for user in user_interaction_cnt if user not in final_selected_users}
        elif su_objective == "sparse":
            su_degree = {user: (1/user_interaction_cnt[user]) for user in user_interaction_cnt if user not in final_selected_users}
        else:
            raise ValueError("Not implemented")
        dem = sum(su_degree.values())
        starting_users = list(set(np.random.choice(list(su_degree.keys()),
                                                   size=su,
                                                   replace=False,
                                                   p=[p / dem for p in su_degree.values()])))

        if len(final_selected_users) + len(starting_users) > total_num_users:  # checking corner cases
            final_selected_users = final_selected_users.union(starting_users[:total_num_users - len(final_selected_users)])
            break
        final_selected_users = final_selected_users.union(starting_users)

        # then selecting their items wrt the objective and choosing a number of them wrt propagation degree
        h0_items = set([l[ITEM_ID_IDX_INTER] for l in interactions if l[USER_ID_IDX_INTER] in starting_users])
        if i_objective == "random":
            h0_items_degree = {item: 1 for item in h0_items}
        elif i_objective == "dense":
            h0_items_degree = {item: item_interaction_cnt[item] for item in h0_items}
        elif i_objective == "sparse":
            h0_items_degree = {item: (1/item_interaction_cnt[item]) for item in h0_items}
        else:
            raise ValueError("Not implemented")
        dem = sum(h0_items_degree.values())
        h0_items_probs = [p / dem for p in h0_items_degree.values()]
        h0_items_keys = list(h0_items_degree.keys())
        print(f"{cnt} items to choose from: {len(h0_items_keys)}")
        chosen_h0_items = list(np.random.choice(h0_items_keys, size=min(item_propagation_number, len(h0_items_keys)),
                                                replace=False, p=h0_items_probs))

        # expanding the chosen items wrt objective, selecting using propagation degree
        h1_users = set([l[USER_ID_IDX_INTER] for l in interactions if l[ITEM_ID_IDX_INTER] in chosen_h0_items])
        # we want to include the initial users, making sure connections
        h1_users = h1_users - final_selected_users  # select new users
        if u_objective == "random":
            h1_users_degree = {u: 1 for u in h1_users}
        elif u_objective == "dense":
            h1_users_degree = {u: user_interaction_cnt[u] for u in h1_users}
        elif u_objective == "sparse":
            h1_users_degree = {u: (1/user_interaction_cnt[u]) for u in h1_users}
        else:
            raise ValueError("Not implemented")
        dem = sum(h1_users_degree.values())
        h1_users_probs = [p / dem for p in h1_users_degree.values()]
        h1_users_keys = list(h1_users_degree.keys())
        # we sample h1 users, and add them to the u0 users
        print(f"{cnt} users to choose from: {len(h1_users_keys)}")
        chosen_h1_users = list(np.random.choice(h1_users_keys, size=min(user_propagation_number, len(h1_users_keys)),
                                                replace=False, p=h1_users_probs))

        if len(final_selected_users) + len(chosen_h1_users) > total_num_users:
            chosen_h1_users = chosen_h1_users[:total_num_users-len(final_selected_users)]
        final_selected_users = final_selected_users.union(chosen_h1_users)

    OUTPUT_DATASET = join(DATASET_PATH,
                          f'total-u-{total_num_users}_'
                          f'{cnt}_rounds_'
                          f'start-u-{starting_num_users}_{su_objective}_'
                          f'item-propag-{item_propagation_number}_{i_objective}_'
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
    parser.add_argument('--su', type=int, default=None, help='starting_num_users')
    parser.add_argument('--item_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--user_p', type=int, default=None, help='item_propagation_number')
    parser.add_argument('--total_users', '-u', type=int, default=None, help='total number of users to sample')
    parser.add_argument('--u_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--i_o', type=str, default=None, help='random/sparse/dense')
    parser.add_argument('--su_o', type=str, default=None, help='random/sparse/dense')
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

    u_objective = args.u_o
    su_objective = args.su_o
    i_objective = args.i_o

    main()
