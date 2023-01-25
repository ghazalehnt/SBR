import argparse
import os
import pickle
import random
from collections import defaultdict
from os.path import join

import pandas as pd
import numpy as np
from datasets import Dataset
from torchtext.data.utils import get_tokenizer
from rank_bm25 import BM25Okapi
import multiprocessing as mp
import time


ITEM_ID_FIELD = "item_id"
USER_ID_FIELD = "user_id"
item_user_inter_text_fields = ["title", "category", "description"]


def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text"):
    samples[f"tokenized_{doc_desc_field}"] = [tokenizer(text) for text in samples[doc_desc_field]]
    return samples


def rank_items(item_id, query):
    if os.path.exists(join(dataset_path, "BM25_item_ranking" ,f"{item_id}.pkl")):
        return
    f = open(join(dataset_path, "BM25_item_ranking" ,f"{item_id}.pkl"), 'wb')
    document_scores = index.get_scores(query)
    document_scores = np.argsort(document_scores)[::-1]
    pickle.dump([doc_ids[doc] for doc in document_scores], f)
    f.close()
    return

if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to dataset')
    args, _ = parser.parse_known_args()
    dataset_path = args.dataset_folder
    os.makedirs(join(dataset_path, "BM25_item_ranking"), exist_ok=True)

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

    start = time.time()
    pool = mp.Pool(mp.cpu_count())
    pool.starmap(rank_items, [(item_id, item_text) for item_id, item_text in zip(item_info["item_id"], item_info["tokenized_text"])])
    pool.close()
    print(f"finished {time.time() - start} seconds")
