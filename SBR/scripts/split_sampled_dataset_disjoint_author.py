import csv
import os
import random
import re
import sys
from collections import defaultdict
from os.path import join
import numpy as np
from sklearn.model_selection import train_test_split
csv.field_size_limit(sys.maxsize)

rating_mapping = {
    '': 0,
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def create_splits(per_user_interactions, ratios, k_core_user, user_author_threshold=3):
    train = []
    test = []
    valid = []
    test_ratio = ratios[1]
    valid_ratio = ratios[2]/(1-test_ratio)  # e.g. [0.6, 0.2, 0.2] -> 0.2 = 0.25 * 0.8
    users_removed_due_to_lack_of_distinct_authors = 0
    users_removed_due_to_k_core = 0
    for user in per_user_interactions:
        if len(per_user_interactions[user].keys()) < user_author_threshold:
            users_removed_due_to_lack_of_distinct_authors += 1
            continue
        X_train, X_test = train_test_split(list(per_user_interactions[user].keys()),
                                           test_size=test_ratio,
                                           random_state=42)
        X_train, X_val = train_test_split(X_train,
                                          test_size=valid_ratio,
                                          random_state=42)
        user_train = []
        for author in X_train:
            user_train.extend(per_user_interactions[user][author])
        user_test = []
        for author in X_test:
            user_test.extend(per_user_interactions[user][author])
        user_val = []
        for author in X_val:
            user_val.extend(per_user_interactions[user][author])

        # creating the k-core
        # the k_core constraint is on all test+val+train >= k-core-user not just train
        if k_core_user is not None:
            if len(user_test) > 0 and len(user_val) > 0 \
                    and (len(user_train) + len(user_val) + len(user_test)) >= k_core_user:
                train.extend(user_train)
                test.extend(user_test)
                valid.extend(user_val)
            else:
                users_removed_due_to_k_core += 1
        else:
            train.extend(user_train)
            test.extend(user_test)
            valid.extend(user_val)

    print(f"users removed due to having fewer than {user_author_threshold} authors: {users_removed_due_to_lack_of_distinct_authors}")
    if k_core_user is not None:
        print(f"users removed due to {k_core_user}-core constraint: {users_removed_due_to_k_core}")

    return train, valid, test


CLEANR = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    #DATASET_PATH = ""
    #INTERACTION_FILE = "goodreads_ucsd.interactions"
    #ITEM_FILE = "goodreads_ucsd.items"
    #USER_FILE = "goodreads_ucsd.users"
    #USER_ID_FIELD = "user_id"
    #ITEM_ID_FIELD = "book_id"
    #RATING_FIELD = "rating"
    #AUTHOR_FIELD = "authors"

    # DATASET_PATH = "TODO"
    # INTERACTION_FILE = "goodreads_crawled.interactions"
    # ITEM_FILE = "goodreads_crawled.items"
    # USER_FILE = "goodreads_crawled.users"
    # USER_ID_FIELD = "user_id"
    # ITEM_ID_FIELD = "item_id"
    # RATING_FIELD = "rating"

    DATASET_PATH = ""
    INTERACTION_FILE = "amazon_reviews_books.interactions"
    ITEM_FILE = "amazon_reviews_books.items"
    USER_FILE = "amazon_reviews_books.users"
    USER_ID_FIELD = "reviewerID"
    ITEM_ID_FIELD = "asin"
    RATING_FIELD = "overall"
    AUTHOR_FIELD = "brand"

    rating_threshold = 4
    user_core_k = 3
    author_case_sensitive = False
    ratio = [0.6, 0.2, 0.2]

    item_authors = defaultdict()
    all_authors = set()
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        ITEM_ID_IDX_ITEM = item_header.index(ITEM_ID_FIELD)
        AUTHOR_IDX_ITEM = item_header.index(AUTHOR_FIELD)
        TITLE_IDX_ITEM = item_header.index("title")
        for line in reader:
            item_id = line[ITEM_ID_IDX_ITEM]
            if not author_case_sensitive:
                author = line[AUTHOR_IDX_ITEM].lower()  # newly added this, it was not lower cased for the www-short
            else:
                author = line[AUTHOR_IDX_ITEM]
            # keeping the author field as it is (do not care about multiple, ...) other meta info
            if author == "":
                continue
            item_authors[item_id] = author
            all_authors.add(author)

    all_interactions_per_user_per_author = defaultdict(lambda: defaultdict(list))
    source_interactions_cnt = 0
    no_author_inters_cnt = 0
    with open(join(DATASET_PATH, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        inter_header = next(reader)
        RATING_IDX_INTER = inter_header.index(RATING_FIELD)
        ITEM_ID_IDX_INTER = inter_header.index(ITEM_ID_FIELD)
        USER_ID_IDX_INTER = inter_header.index(USER_ID_FIELD)
        for line in reader:
            item_id = line[ITEM_ID_IDX_INTER]
            user_id = line[USER_ID_IDX_INTER]
            if rating_threshold is not None:
                if INTERACTION_FILE.startswith("goodreads_crawled"):
                    rating = rating_mapping[line[RATING_IDX_INTER]]
                elif INTERACTION_FILE.startswith("goodreads_ucsd"):
                    rating = int(line[RATING_IDX_INTER])
                elif INTERACTION_FILE.startswith("amazon_reviews_books"):
                    rating = int(float(line[RATING_IDX_INTER]))
                else:
                    raise NotImplementedError("not implemented!")
                if rating < rating_threshold:
                    continue
            if item_id not in item_authors:
                no_author_inters_cnt += 1
                continue
            all_interactions_per_user_per_author[user_id][item_authors[item_id]].append(line)
            source_interactions_cnt += 1
    print(f"{no_author_inters_cnt} interactions with no author removed")
    print(f"{source_interactions_cnt} total interactions to choose from")

    # idea is to split by author per user, then add all the chosen author books to the set
    train_set, test_set, valid_set = create_splits(all_interactions_per_user_per_author, ratio, user_core_k)

    out_path = join(DATASET_PATH,
                    f"disjoint-auth-cs-{author_case_sensitive}_rating-th-{rating_threshold}"
                    f"_user-core-{user_core_k if user_core_k is not None else 'None'}"
                    f"_ratios{'-'.join([str(r) for r in ratio])}")
    os.makedirs(out_path, exist_ok=True)

    inter_header[USER_ID_IDX_INTER] = "user_id"
    inter_header[ITEM_ID_IDX_INTER] = "item_id"
    inter_header[RATING_IDX_INTER] = "rating"
    with open(join(out_path, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in train_set])

    # only keep user_id, item_id, rating
    temp = []
    for line in valid_set:
        temp.append([re.sub(CLEANR, '', line[USER_ID_IDX_INTER]),
                     re.sub(CLEANR, '', line[ITEM_ID_IDX_INTER]),
                     re.sub(CLEANR, '', line[RATING_IDX_INTER])])
    with open(join(out_path, "validation.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "rating"])
        writer.writerows(temp)

    temp = []
    for line in test_set:
        temp.append([re.sub(CLEANR, '', line[USER_ID_IDX_INTER]),
                     re.sub(CLEANR, '', line[ITEM_ID_IDX_INTER]),
                     re.sub(CLEANR, '', line[RATING_IDX_INTER])])
    with open(join(out_path, "test.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "item_id", "rating"])
        writer.writerows(temp)

    all_users = [line[USER_ID_IDX_INTER] for line in train_set]
    all_users.extend([line[USER_ID_IDX_INTER] for line in test_set])
    all_users.extend([line[USER_ID_IDX_INTER] for line in valid_set])
    all_users = set(all_users)

    all_items = [line[ITEM_ID_IDX_INTER] for line in train_set]
    all_items.extend([line[ITEM_ID_IDX_INTER] for line in test_set])
    all_items.extend([line[ITEM_ID_IDX_INTER] for line in valid_set])
    all_items = set(all_items)

    # copy user and item files, change header, keep only users/items in the dataset
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as fin, open(join(out_path, "items.csv"), 'w') as fout:
        reader = csv.reader(fin)
        item_header = next(reader)
        writer = csv.writer(fout)
        item_field_idx_item = item_header.index(ITEM_ID_FIELD)
        item_header[item_field_idx_item] = "item_id"
        writer.writerow(item_header)
        for line in reader:
            if line[item_field_idx_item] not in all_items:
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
