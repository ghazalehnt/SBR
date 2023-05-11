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

def tokenize_function_torchtext(samples, tokenizer=None, doc_desc_field="text"):
    samples[f"tokenized_{doc_desc_field}"] = [tokenizer(text) for text in samples[doc_desc_field]]
    return samples


def rank_items(item_id, query):
    if os.path.exists(join(dataset_path, "BM25_item_ranking_filtered_by_tags", f"{item_id}.pkl")):
        return
    f = open(join(dataset_path, "BM25_item_ranking_filtered_by_tags", f"{item_id}.pkl"), 'wb')
    candidate_doc_ids = []
    for tag in item_tags[item_id]:
        candidate_doc_ids.extend(tag_items[tag])

    candidate_doc_ids = list(set(candidate_doc_ids))
    candidate_internal_ids = [doc_ids[c_doc_id] for c_doc_id in candidate_doc_ids]
    document_scores = index.get_batch_scores(query, candidate_internal_ids)
    document_scores_idx = np.argsort(document_scores)[::-1]
    pickle.dump([candidate_doc_ids[doc] for doc in document_scores_idx[:topn]], f)
    f.close()
    return


def get_test_items(dataset_path):
    test_dataset = pd.read_csv(os.path.join(dataset_path, f"test.csv"), dtype=str)
    return set(test_dataset["item_id"])


def get_items_by_genre():
    ret_genre_items = defaultdict(list)
    ret_item_genres = defaultdict(list)
    # some book do not  have any genre, these are considered as same genre! as we don't want to loose them in neg sampling
    for item_id, genres in zip(item_info[ITEM_ID_FIELD], item_info[tag_field]):
        if tag_field in ["category", "genres"]:
            for g in [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]:
                if g == "Books":
                    continue
                ret_genre_items[g].append(item_id)
                ret_item_genres[item_id].append(g)
        else:
            raise NotImplementedError()
    return ret_genre_items, ret_item_genres


if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', type=str, help='path to dataset')
    parser.add_argument('--topn', type=int, default=1000, help='path to dataset')
    parser.add_argument('--tag_field', type=str, help='tag_field')
    parser.add_argument('--query_fields', type=str, nargs='+', help='query_fields')

    args, _ = parser.parse_known_args()
    dataset_path = args.dataset_folder
    os.makedirs(join(dataset_path, "BM25_item_ranking_filtered_by_tags"), exist_ok=True)
    topn = args.topn
    tag_field = args.tag_field
    query_fields = args.query_fields

    # test-items to be considered query:
    to_calc_items = get_test_items(dataset_path)
    use_col = [ITEM_ID_FIELD, tag_field]
    use_col.extend(query_fields)
    item_info = pd.read_csv(os.path.join(dataset_path, "items.csv"), dtype=str, usecols=use_col)
    item_info = item_info.fillna("")
    print("load item meta data")
    tag_items, item_tags = get_items_by_genre()
    print("genre items loaded")
    for col in query_fields:
        item_info[col] = item_info[col].apply(
            lambda x: ", ".join(
                x.replace("[", "").replace("]", "").replace("'", "").replace('"', "").replace("  ", " ").split(","))
        )
    item_info['text'] = item_info[query_fields].agg('. '.join, axis=1)
    item_info = item_info.drop(columns=query_fields)
    tokenizer = get_tokenizer("basic_english")
    item_info = Dataset.from_pandas(item_info)
    item_info = item_info.map(tokenize_function_torchtext, batched=True,
                              fn_kwargs={"tokenizer": tokenizer, "doc_desc_field": 'text'})
    item_info = item_info.remove_columns(['text'])
    item_info = item_info.to_pandas()

    # pool of items: indexed:
    index = BM25Okapi(item_info['tokenized_text'])
    doc_ids = {item_id: i for item_id, i in zip(item_info[ITEM_ID_FIELD], range(len(item_info[ITEM_ID_FIELD])))}
    print("BM25 index created")

    start = time.time()
    pool = mp.Pool(mp.cpu_count())

    shuffled_list = [(item_id, item_text) for item_id, item_text in
                     zip(item_info["item_id"], item_info["tokenized_text"]) if item_id in to_calc_items]
    random.shuffle(shuffled_list)

    pool.starmap(rank_items, [item for item in shuffled_list])
    pool.close()
    print(f"finished {time.time() - start} seconds")
