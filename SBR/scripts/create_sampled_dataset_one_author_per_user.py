import csv
import math
import os
import random
import re
from collections import defaultdict
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

    # DATASET_PATH = ".../extracted_dataset..."
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
    TIE_BREAK_FIELD = "rank"

    rating_threshold = 3

    out_path = join(DATASET_PATH, f"one_author_per_user_dataset_rating_threshold{rating_threshold}")
    os.makedirs(out_path, exist_ok=True)

    item_authors = defaultdict()
    all_authors = set()
    if INTERACTION_FILE.startswith("amazon_reviews_books"):
        item_tie_breaks = defaultdict(lambda: {0: math.inf, 1: math.inf, 2: math.inf, 3: math.inf})
    else:
        raise NotImplementedError("todo")
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        ITEM_ID_IDX_ITEM = item_header.index(ITEM_ID_FIELD)
        AUTHOR_IDX_ITEM = item_header.index(AUTHOR_FIELD)
        TITLE_IDX_ITEM = item_header.index("title")
        TIE_BREAK_IDX_ITEM = item_header.index(TIE_BREAK_FIELD)
        for line in reader:
            item_id = line[ITEM_ID_IDX_ITEM]
            author = line[AUTHOR_IDX_ITEM]
            # keeping the author field as it is (do not care about multiple, ...) other meta info
            if author == "":
                continue
            item_authors[item_id] = author
            tie_brk = line[TIE_BREAK_IDX_ITEM]
            # todo adapt for goodreads for avg rating
            if tie_brk in ["", "[]"]:
                tie_brk = 0
            elif "in" in tie_brk:
                # in Books
                try:
                    tie_brk = int(re.sub(r"(?:\['>#)?([\d,]+) in Books.*", '\g<1>', tie_brk).replace(',', ''))
                    item_tie_breaks[item_id][0] = tie_brk
                except:
                    try:
                        tie_brk = int(re.sub(r"(?:\['>#)?([\d,]+) Paid in Kindle Store.*", '\g<1>', tie_brk).replace(',', ''))
                        item_tie_breaks[item_id][1] = tie_brk
                    except:
                        try:
                            tie_brk = int(
                                re.sub(r"(?:\['>#)?([\d,]+) Free in Kindle Store.*", '\g<1>', tie_brk).replace(',', ''))
                            item_tie_breaks[item_id][2] = tie_brk
                        except:
                            try:
                                tie_brk = int(
                                    re.sub(r"(?:\['>#)?([\d,]+) in .*", '\g<1>', tie_brk).replace(',',
                                                                                                                   ''))
                                item_tie_breaks[item_id][3] = tie_brk
                            except:
                                print(tie_brk)
                                tie_brk = 0
            else:
                print(tie_brk)
                tie_brk = 0
            all_authors.add(author)

    all_interactions_per_user_per_author = defaultdict(lambda : defaultdict(list))
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
            if item_id not in item_authors:
                no_author_inters_cnt += 1
                continue
            if rating_threshold is not None:
                if INTERACTION_FILE.startswith("goodreads_crawled"):
                    rating = rating_mapping[line[RATING_IDX_INTER]]
                elif INTERACTION_FILE.startswith("amazon_reviews_books"):
                    rating = int(float(line[RATING_IDX_INTER]))
                else:
                    raise NotImplementedError("not implemented!")
                if rating < rating_threshold:
                    continue
            all_interactions_per_user_per_author[user_id][item_authors[item_id]].append(line)
            source_interactions_cnt += 1
    print(f"{no_author_inters_cnt} interactions with no author removed")
    print(f"{source_interactions_cnt} total interactions to choose from")

    interactions = []
    remaining_items = set()
    for user, author_inters in all_interactions_per_user_per_author.items():
        for author, inters in author_inters.items():
            if len(inters) == 0:
                raise ValueError(f"user {user}, author {author} had no interactions!?")
            if len(inters) == 1:
                interactions.append(inters[0])
                remaining_items.add(inters[0][ITEM_ID_IDX_INTER])
            else:
                # choosing 1 book per author per user
                inters = sorted(inters, key=lambda x: (-1 * int(float(x[RATING_IDX_INTER])),
                                                       item_tie_breaks[x[ITEM_ID_IDX_INTER]][0],
                                                       item_tie_breaks[x[ITEM_ID_IDX_INTER]][1],
                                                       item_tie_breaks[x[ITEM_ID_IDX_INTER]][2],
                                                       item_tie_breaks[x[ITEM_ID_IDX_INTER]][3]),
                                reverse=False)  # main sort is rating, bigger the better. but we *-1 to be coherent with other tie breakers which is the rank, smaller better.
                interactions.append(inters[0])
                remaining_items.add(inters[0][ITEM_ID_IDX_INTER])

    print(f"remaining items: {len(remaining_items)}")

    # write down
    # read and write users which are all users
    with open(join(DATASET_PATH, USER_FILE), 'r') as fin, open(join(out_path, USER_FILE), 'w') as fout:
        reader = csv.reader(fin)
        user_header = next(reader)
        writer = csv.writer(fout)
        writer.writerow(user_header)
        for line in reader:
            writer.writerow(line)

    # write items:
    with open(join(DATASET_PATH, ITEM_FILE), 'r') as fin, open(join(out_path, ITEM_FILE), 'w') as fout:
        reader = csv.reader(fin)
        item_header = next(reader)
        writer = csv.writer(fout)
        writer.writerow(item_header)
        for line in reader:
            if line[ITEM_ID_IDX_ITEM] in remaining_items:
                writer.writerow(line)

    # write interactions:
    with open(join(out_path, INTERACTION_FILE), 'w') as fout:
        writer = csv.writer(fout)
        writer.writerow(inter_header)
        writer.writerows(interactions)
    print(f"remaining interactions: {len(interactions)}")

    print("Done")

    # TODOthere are cases where if we remove Visit ... from the author name it exists in the authores.. but let's keep it now as is (124 out of ~1.4M authors though!)
    # for author in all_authors:
    #     if "Visit Amazon" in author:
    #         temp = re.sub(r"Visit Amazon's (.*) Page", r"\g<1>", author)
    #         if temp in all_authors:
    #             print(author)
    #             print(temp)
    # example
    # for user, inters in all_interactions_per_user_per_author.items():
    #     if "Visit Amazon's RH Disney Page" in inters.keys():
    #         print(inters["Visit Amazon's RH Disney Page"])
    #         break
