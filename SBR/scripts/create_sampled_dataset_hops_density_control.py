import csv
import os
import random
from os.path import join
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)

rating_mapping = {
    '': 0,
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def get_per_user_interaction_cnt(inters, USER_ID_IDX):
    ret = {}
    for line in inters:
        user_id = line[USER_ID_IDX]
        if user_id not in ret:
            ret[user_id] = 0
        ret[user_id] += 1
    return ret


#  Here we randomly select N users, then expand them by max 2 hops.
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    #DATASET_PATH = ".../extracted_dataset..."
    #INTERACTION_FILE = "goodreads_crawled.interactions"
    #ITEM_FILE = "goodreads_crawled.items"
    #USER_FILE = "goodreads_crawled.users"
    #USER_ID_FIELD = "user_id"
    #ITEM_ID_FIELD = "item_id"
    #RATING_FIELD = "rating"
    
    DATASET_PATH = "tODO"
    INTERACTION_FILE = "amazon_reviews_books.interactions"
    ITEM_FILE = "amazon_reviews_books.items"
    USER_FILE = "amazon_reviews_books.users"
    USER_ID_FIELD = "reviewerID"
    ITEM_ID_FIELD = "asin"
    RATING_FIELD = "overall"

    rating_threshold = 3
    starting_num_users = 5
    num_lt_users = 5
    num_h0_items = 10
    total_num_users = 50
    #objective = 'dense'
    #objective = 'sparse'
    objective = 'random'
    LT_THRESHOLD = 4

    interactions = []
    with open(join(DATASET_PATH, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        inter_header = next(reader)
        RATING_IDX_INTER = inter_header.index(RATING_FIELD)
        for line in reader:
            if rating_threshold is not None:
                if INTERACTION_FILE.startswith("goodreads_crawled"):
                    rating = rating_mapping[line[RATING_IDX_INTER]]
                elif INTERACTION_FILE.startswith("amazon_reviews_books"):
#                    if len(line) < RATING_IDX_INTER:
#                    print(line)
                    rating = int(float(line[RATING_IDX_INTER]))
                else:
                    raise NotImplementedError("not implemented!")
                if rating < rating_threshold:
                    continue
            interactions.append(line)
    print(inter_header)

    USER_ID_IDX_INTER = inter_header.index(USER_ID_FIELD)
    ITEM_ID_IDX_INTER = inter_header.index(ITEM_ID_FIELD)

    user_interaction_cnt = get_per_user_interaction_cnt(interactions, USER_ID_IDX_INTER)

    lt_users = [user for user in user_interaction_cnt.keys() if user_interaction_cnt[user] <= LT_THRESHOLD]
    shared_users = user_interaction_cnt.keys() - set(lt_users)

    chosen_long_tail_users = random.sample(lt_users, k=num_lt_users)
    chosen_users = random.sample(shared_users, k=starting_num_users)
    h0_users = set(chosen_users).union(chosen_long_tail_users)

    # grow chosen users by 2 hops
    h0_items = set([l[ITEM_ID_IDX_INTER] for l in interactions if l[USER_ID_IDX_INTER] in h0_users])
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
            h0_items_degree = {k: (1/v) for k, v in h0_items_degree.items()}
        else:
            raise ValueError("Not implemented")
    dem = sum(h0_items_degree.values())
    h0_items_probs = [p/dem for p in h0_items_degree.values()]
    h0_items_keys = list(h0_items_degree.keys())
    print(len(h0_items_keys))
    chosen_h0_items = list(np.random.choice(h0_items_keys, size=num_h0_items, replace=False, p=h0_items_probs))

    h1_users = set([l[USER_ID_IDX_INTER] for l in interactions if l[ITEM_ID_IDX_INTER] in chosen_h0_items])
    # we want to include the initial users, making sure connections
    h1_users = h1_users - h0_users

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

    OUTPUT_DATASET = join(DATASET_PATH,
                          f'sampled_dataset_threshold{rating_threshold}_totalu{total_num_users}_'
                          f'su{starting_num_users}_ltth{LT_THRESHOLD}_sltu{num_lt_users}_h1i{num_h0_items}_{objective}')
    os.makedirs(OUTPUT_DATASET, exist_ok=True)

    total_items = set()
    total_users = set(all_users)
    with open(join(OUTPUT_DATASET, INTERACTION_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        temp = []
        for line in interactions:
            user_id = line[USER_ID_IDX_INTER]
            item_id = line[ITEM_ID_IDX_INTER]
            if user_id in all_users:
                total_items.add(item_id)
                temp.append(line)
        writer.writerows(temp)

    # read/write user and item files:
    user_info = []
    with open(join(DATASET_PATH, USER_FILE), 'r') as f:
        reader = csv.reader(f)
        user_header = next(reader)
        IDX = user_header.index(USER_ID_FIELD)
        for line in reader:
            if line[IDX] in total_users:
                user_info.append(line)
    item_info = []
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        IDX = item_header.index(ITEM_ID_FIELD)
        for line in reader:
            if line[IDX] in total_items:
                item_info.append(line)

    with open(join(OUTPUT_DATASET, USER_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(user_header)
        writer.writerows(user_info)

    with open(join(OUTPUT_DATASET, ITEM_FILE), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(item_header)
        writer.writerows(item_info)
