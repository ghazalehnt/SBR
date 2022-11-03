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


def create_splits(per_item_interactions, ratios, longtail_trainonly_th):
    train = []
    valid = []
    valid_ratio = ratios[1]
    for item in per_item_interactions:
        if len(per_item_interactions[item]) <= longtail_trainonly_th:
            for line in per_item_interactions[item]:
                train.append(line)
        else:
            X_train, X_val = train_test_split(per_item_interactions[item],
                                                                test_size=valid_ratio,
                                                                random_state=42)
            for line in X_train:
                train.append(line)
            for line in X_val:
                valid.append(line)

    return train, valid


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    DATASET_PATH = "TODO"
    # INTERACTION_FILE = "goodreads_crawled.interactions"
    # ITEM_FILE = "goodreads_crawled.items"
    # USER_FILE = "goodreads_crawled.users"
    # USER_ID_FIELD = "user_id"
    # ITEM_ID_FIELD = "item_id"
    # RATING_FIELD = "rating"
    INTERACTION_FILE = "amazon_reviews_books.interactions"
    ITEM_FILE = "amazon_reviews_books.items"
    USER_FILE = "amazon_reviews_books.users"
    USER_ID_FIELD = "reviewerID"
    ITEM_ID_FIELD = "asin"
    RATING_FIELD = "overall"

    lt_threshold = 4
    ratios = [0.8, 0.2]

    # read items first, to ommit the interactions whose item does not exist in the items meta data: (TODO should do it one level higher when transforming the raw data, but for now here is good)
    item_metadata_ids = set()
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as fin:
        reader = csv.reader(fin)
        item_header = next(reader)
        item_id_idx = item_header.index(ITEM_ID_FIELD)
        for line in reader:
            item_metadata_ids.add(line[item_id_idx])

    per_item_interactions = defaultdict(list)
    with open(join(DATASET_PATH, INTERACTION_FILE), 'r') as f:
        reader = csv.reader(f)
        inter_header = next(reader)
        USER_ID_IDX_INTER = inter_header.index(USER_ID_FIELD)
        ITEM_ID_IDX_INTER = inter_header.index(ITEM_ID_FIELD)
        for line in reader:
            if line[ITEM_ID_IDX_INTER] not in item_metadata_ids:
                continue
            per_item_interactions[line[ITEM_ID_IDX_INTER]].append(line)

    train, valid = create_splits(per_item_interactions, ratios, lt_threshold)

    out_path = join(DATASET_PATH, f"ltth{lt_threshold}_ratios{'-'.join([str(r) for r in ratios])}_per_item")
    os.makedirs(out_path, exist_ok=True)

    inter_header[USER_ID_IDX_INTER] = "user_id"
    inter_header[ITEM_ID_IDX_INTER] = "item_id"
    inter_header[inter_header.index(RATING_FIELD)] = "rating"
    with open(join(out_path, "train.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in train])
    with open(join(out_path, "validation.csv"), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(inter_header)
        writer.writerows([[re.sub(CLEANR, '', l) for l in line] for line in valid])

    # copy user and item files, change header
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as fin, open(join(out_path, "items.csv"), 'w') as fout:
        reader = csv.reader(fin)
        item_header = next(reader)
        writer = csv.writer(fout)
        item_header[item_header.index(ITEM_ID_FIELD)] = "item_id"
        writer.writerow(item_header)
        for line in reader:
            # try to clean the genres
            if ITEM_FILE.startswith("goodreads_crawled"):
                if "like" in line[item_header.index("genres")]:
                    line[item_header.index("genres")] = ''
            writer.writerow([re.sub(CLEANR, '', l) for l in line])

    with open(join(DATASET_PATH, USER_FILE), 'r') as fin, open(join(out_path, "users.csv"), 'w') as fout:
        reader = csv.reader(fin)
        user_header = next(reader)
        writer = csv.writer(fout)
        user_header[user_header.index(USER_ID_FIELD)] = "user_id"
        writer.writerow(user_header)
        for line in reader:
            writer.writerow([re.sub(CLEANR, '', l) for l in line])

    allusers = set([line[USER_ID_IDX_INTER] for line in train])
    allusers = allusers.union(set([line[USER_ID_IDX_INTER] for line in valid]))
    print(f"num users: {len(allusers)}")

    allitems = set([line[ITEM_ID_IDX_INTER] for line in train])
    allitems = allitems.union(set([line[ITEM_ID_IDX_INTER] for line in valid]))
    print(f"num items: {len(allitems)}")
