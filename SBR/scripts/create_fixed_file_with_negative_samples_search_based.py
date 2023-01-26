import argparse
import csv
import os
import pickle
import random
from collections import  defaultdict
import time

import pandas as pd

ITEM_ID_FIELD = "item_id"
USER_ID_FIELD = "user_id"
num_retrieved_items = 1000

def load_data(dataset_path):
    ret = defaultdict()
    for sp in ["train", "validation", "test"]:
        ret[sp] = pd.read_csv(os.path.join(dataset_path, f"{sp}.csv"), dtype=str)
    return ret


def get_user_used_items_for_test(dataset_path):
    datasets = load_data(dataset_path)
    used_items = defaultdict(lambda: defaultdict(set))
    for split in datasets.keys():
        for user_iid, item_iid in zip(datasets[split][USER_ID_FIELD], datasets[split][ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)
    # for test: train, valid, and current positive items are used items.
    ret = used_items['train'].copy()
    for user_id, cur_user_items in used_items['validation'].items():
        ret[user_id] = ret[user_id].union(cur_user_items)
    for user_id, cur_user_items in used_items['test'].items():
        ret[user_id] = ret[user_id].union(cur_user_items)
    return ret, datasets['test']


def main(dataset_path, num_neg_samples):
    used_items, test_dataset = get_user_used_items_for_test(dataset_path)
    print("load test dataset")
    bm25_folder = os.path.join(dataset_path, "BM25_item_ranking")

    item_bm25_ranking = {}
    for item_id in set(test_dataset["item_id"]):
        item_bm25_ranking[item_id] = pickle.load(open(os.path.join(bm25_folder, f"{item_id}.pkl"), 'rb'))

    f = open(os.path.join(dataset_path, f'test_neg_SB_BM25_{num_neg_samples}.csv'), 'w')
    writer = csv.writer(f)
    writer.writerow([USER_ID_FIELD, ITEM_ID_FIELD, 'label', 'ref_item'])

    cnt = 0
    start = time.time()
    for item_id, user_id in zip(test_dataset[ITEM_ID_FIELD], test_dataset[USER_ID_FIELD]):
        neg_samples = []
        for doc_id in item_bm25_ranking[item_id][:num_retrieved_items]:
            if len(neg_samples) == num_neg_samples:
                break
            if doc_id == item_id:
                continue
            if doc_id in used_items[user_id]:
                continue
            neg_samples.append(doc_id)
        if len(neg_samples) < 100:
            print(f"could not sample 100 negs for item: {item_id} user: {user_id}")
        writer.writerows([[user_id, sampled_item_id, 0, item_id] for sampled_item_id in neg_samples])
        cnt += 1
        if cnt % 1000 == 0:
            print(f"{cnt} done!")
            print(f"is {time.time() - start} seconds")
            f.flush()
    f.close()
    print("test done")


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to dataset')
    parser.add_argument('--ns', type=int, help='number of negative samples')
    args, _ = parser.parse_known_args()

    main(args.dataset_folder, args.ns)