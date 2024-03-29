import csv
import os
import random
import re
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
import sys
csv.field_size_limit(sys.maxsize)

CLEANR = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

goodreads_rating_mapping = {
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def create_splits(per_user_interactions, ratios, user_longtail_trainonly_th, user_keep_the_longrail):
    train = []
    test = []
    valid = []
    test_ratio = ratios[1]
    valid_ratio = ratios[2]/(1-test_ratio)  # e.g. [0.6, 0.2, 0.2] -> 0.2 = 0.25 * 0.8
    for user in per_user_interactions:
        if len(per_user_interactions[user]) <= user_longtail_trainonly_th:
            if user_keep_the_longrail:
                for line in per_user_interactions[user]:
                    train.append(line)
        else:
            X_train, X_test = train_test_split(per_user_interactions[user],
                                                                test_size=test_ratio,
                                                                random_state=42)

            X_train, X_val = train_test_split(X_train,
                                                              test_size=valid_ratio,
                                                              random_state=42)
            for line in X_train:
                train.append(line)
            for line in X_test:
                test.append(line)
            for line in X_val:
                valid.append(line)

    return train, valid, test


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    DATASET_PATH = "TODO"
    INTERACTION_FILE = "goodreads_ucsd.interactions"
    ITEM_FILE = "goodreads_ucsd.items"
    USER_FILE = "goodreads_ucsd.users"
    USER_ID_FIELD = "user_id"
    ITEM_ID_FIELD = "book_id"
    RATING_FIELD = "rating"

   # DATASET_PATH = "TODO"
   # INTERACTION_FILE = "goodreads_crawled.interactions"
   # ITEM_FILE = "goodreads_crawled.items"
   # USER_FILE = "goodreads_crawled.users"
   # USER_ID_FIELD = "user_id"
   # ITEM_ID_FIELD = "item_id"
   # RATING_FIELD = "rating"

   # DATASET_PATH = "TODO"
   # INTERACTION_FILE = "amazon_reviews_books.interactions"
   # ITEM_FILE = "amazon_reviews_books.items"
   # USER_FILE = "amazon_reviews_books.users"
   # USER_ID_FIELD = "reviewerID"
   # ITEM_ID_FIELD = "asin"
   # RATING_FIELD = "overall"

    rating_threshold = 4
    keep_lt_users = True
    user_lt_threshold = 4
    ratios = [0.6, 0.2, 0.2]

    # read items, to only keep items from interactions that exist in item-meta:
    item_meta_info = defaultdict()
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as fin:
        reader = csv.reader(fin)
        item_header = next(reader)
        item_field_idx_item = item_header.index(ITEM_ID_FIELD)
        item_header[item_field_idx_item] = "item_id"
        for line in reader:
            item_meta_info[line[item_field_idx_item]] = line

    per_user_interactions = defaultdict(list)
    with open(join(DATASET_PATH, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        inter_header = next(reader)
        USER_ID_IDX_INTER = inter_header.index(USER_ID_FIELD)
        ITEM_ID_IDX_INTER = inter_header.index(ITEM_ID_FIELD)
        RATING_IDX_INTER = inter_header.index(RATING_FIELD)
        for line in reader:
            if line[ITEM_ID_IDX_INTER] not in item_meta_info.keys():
                continue
            if rating_threshold is not None:
                if INTERACTION_FILE.startswith("goodreads_crawled"):
                    if goodreads_rating_mapping[line[RATING_IDX_INTER]] < rating_threshold:
                        continue
                else:
                    if int(float(line[RATING_IDX_INTER])) < rating_threshold:
                        continue
            per_user_interactions[line[USER_ID_IDX_INTER]].append(line)

    train, valid, test = create_splits(per_user_interactions, ratios,
                                       user_lt_threshold, keep_lt_users)

    out_path = join(DATASET_PATH,
                    f"rating-th-{rating_threshold if rating_threshold is not None else 'None'}"
                    f"_u-ltth{user_lt_threshold}-{'kept' if keep_lt_users else 'dropped'}"
                    f"_ratios{'-'.join([str(r) for r in ratios])}")
    os.makedirs(out_path, exist_ok=True)

    inter_header[USER_ID_IDX_INTER] = "user_id"
    inter_header[ITEM_ID_IDX_INTER] = "item_id"
    inter_header[RATING_IDX_INTER] = "rating"
    with open(join(out_path, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in train])
    with open(join(out_path, "validation.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in valid])
    with open(join(out_path, "test.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in test])

    all_users = [line[USER_ID_IDX_INTER] for line in train]
    all_users.extend([line[USER_ID_IDX_INTER] for line in test])
    all_users.extend([line[USER_ID_IDX_INTER] for line in valid])
    all_users = set(all_users)

    all_items = [line[ITEM_ID_IDX_INTER] for line in train]
    all_items.extend([line[ITEM_ID_IDX_INTER] for line in test])
    all_items.extend([line[ITEM_ID_IDX_INTER] for line in valid])
    all_items = set(all_items)

    # copy user and item files, change header, keep only users/items in the dataset
    with open(join(out_path, "items.csv"), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(item_header)
        for item_id, line in item_meta_info.items():
            if item_id not in all_items:
                continue
            # try to clean the genres
            if ITEM_FILE.startswith("goodreads_crawled"):
                if "like" in line[item_header.index("genres")]:
                    line[item_header.index("genres")] = ''
            writer.writerow([re.sub(CLEANR, '', l) for l in line])

    with open(join(DATASET_PATH, USER_FILE), 'r') as fin, open(join(out_path, "users.csv"), 'w') as fout:
        reader = csv.reader(fin)
        user_header = next(reader)
        writer = csv.writer(fout)
        user_field_idx_user = user_header.index(USER_ID_FIELD)
        user_header[user_field_idx_user] = "user_id"
        writer.writerow(user_header)
        for line in reader:
            if line[user_field_idx_user] not in all_users:
                continue
            writer.writerow([re.sub(CLEANR, '', l) for l in line])

    print(f"num users: {len(all_users)}")
    print(f"num items: {len(all_items)}")
