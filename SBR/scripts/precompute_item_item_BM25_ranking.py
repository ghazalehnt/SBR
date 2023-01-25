import argparse
import json
import os
import random
from collections import defaultdict
from os.path import join

import pandas as pd
import numpy as np
from datasets import Dataset
from torchtext.data.utils import get_tokenizer
from rank_bm25 import BM25Okapi
import multiprocessing as mp


ITEM_ID_FIELD = "item_id"
USER_ID_FIELD = "user_id"
item_user_inter_text_fields = ["title", "category", "description"]


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


def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text"):
    samples[f"tokenized_{doc_desc_field}"] = [tokenizer(text) for text in samples[doc_desc_field]]
    return samples


def rank_items(query):
    document_scores = index.get_scores(query)
    document_scores = np.argsort(document_scores)[::-1]
    return document_scores


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to dataset')
    args, _ = parser.parse_known_args()
    dataset_path = args.dataset_folder

    use_col = [ITEM_ID_FIELD]
    use_col.extend(item_user_inter_text_fields)
    item_info = pd.read_csv(os.path.join(dataset_path, "items.csv"), dtype=str, usecols=use_col)
    item_info = item_info.fillna("")
    print("load item meta data")
    for col in item_user_inter_text_fields:
        item_info[col] = item_info[col].apply(
            lambda x: ", ".join(
                x.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("  ", " ").split(","))
        )
    item_info['text'] = item_info[item_user_inter_text_fields].agg('. '.join, axis=1)
    item_info = item_info.drop(columns=item_user_inter_text_fields)
    tokenizer = get_tokenizer("basic_english")
    item_info = Dataset.from_pandas(item_info)
    item_info = item_info.map(tokenize_function_torchtext, batched=True,
                              fn_kwargs={"tokenizer": tokenizer, "doc_desc_field": 'text'})
    item_info = item_info.remove_columns(['text'])
    item_info = item_info.to_pandas()

    index = BM25Okapi(item_info['tokenized_text'])
    doc_ids = list(item_info[ITEM_ID_FIELD])
    print("BM25 index created")

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(rank_items, [item_text for item_text in item_info["tokenized_text"]])
    pool.close()

    item_item_ranking = {}
    for item_id, r in zip(item_info["item_id"], results):
        item_item_ranking[item_id] = r

    with open(join(dataset_path, "items_BM25_ranking_per_item.json"), 'w') as f:
        json.dump(item_item_ranking, f)