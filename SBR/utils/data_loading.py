import json
import random
import time
from builtins import NotImplementedError
from collections import Counter, defaultdict
from os.path import join

import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict
from sentence_splitter import SentenceSplitter
from sentence_transformers import SentenceTransformer, util
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.filter_user_profile import filter_user_profile_idf_sentences, filter_user_profile_idf_tf

goodreads_rating_mapping = {
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def tokenize_function(examples, tokenizer, field, max_length, max_num_chunks):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        padding="max_length"  # we pad the chunks here, because it would be too complicated later due to the chunks themselves...
    )

    sample_map = result.pop("overflow_to_sample_mapping")
    # Here they expand other fields of the data to match the number of chunks... repeating user id ...
    # this creates new examples, is it something we want? IDK maybe
    # for key, values in examples.items():
    #     result[key] = [values[i] for i in sample_map]
    # # todo here maybe have a max chunk thing?
    # Instead we could have a field having all chunks:
    examples['chunks_input_ids'] = [[] for i in range(len(examples[field]))]
    examples['chunks_attention_mask'] = [[] for i in range(len(examples[field]))]
    for i, j in zip(sample_map, range(len(result['input_ids']))):
        if max_num_chunks is None or len(examples['chunks_input_ids'][i]) < max_num_chunks:
            examples['chunks_input_ids'][i].append(result['input_ids'][j])
            examples['chunks_attention_mask'][i].append(result['attention_mask'][j])
    return examples


# def sentencize_function(samples, sentencizer=None, doc_desc_field="text",
#                         case_sensitive=True, normalize_negation=True):
#     sent_ret = []
#
#     for text in samples[doc_desc_field]:
#         sents = []
#         for s in sentencizer.split(text=text):
#             if not case_sensitive:
#                 s = s.lower()
#             if normalize_negation:
#                 s = s.replace("n't", " not")
#             sents.append(s)
#         sent_ret.append(sents)
#     return {f"sentences_{doc_desc_field}": sent_ret}


def sentencize(text, sentencizer, case_sensitive, normalize_negation):
    sents = []
    for s in sentencizer.split(text=text):
        if not case_sensitive:
            s = s.lower()
        if normalize_negation:
            s = s.replace("n't", " not")
        sents.append(s)
    return sents


def filter_user_profile(dataset_config, user_info):
    # filtering the user profile
    # filter-type1.1 idf_sentence
    if dataset_config['user_text_filter'] == "idf_sentence":
        user_info = filter_user_profile_idf_sentences(dataset_config, user_info)
    # filter-type1 idf: we can have idf_1_all, idf_2_all, idf_3_all, idf_1-2_all, ..., idf_1-2-3_all, idf_1_unique, ...
    # filter-type2 tf-idf: tf-idf_1, ..., tf-idf_1-2-3
    elif dataset_config['user_text_filter'].startswith("idf_") or \
            dataset_config['user_text_filter'].startswith("tf-idf_"):
        user_info = filter_user_profile_idf_tf(dataset_config, user_info)
    else:
        raise ValueError(
            f"filtering method not implemented, or belong to another script! {dataset_config['user_text_filter']}")

    return Dataset.from_pandas(user_info, preserve_index=False)


def load_data(config, pretrained_model):
    start = time.time()
    print("Start: load dataset...")
    if 'user_text_filter' in config and config['user_text_filter'] in ["idf_sentence"]:
        temp_cs = config['case_sensitive']
        config['case_sensitive'] = True
    datasets, user_info, item_info, filtered_out_user_item_pairs_by_limit = load_split_dataset(config)
    if 'user_text_filter' in config and config['user_text_filter'] in ["idf_sentence"]:
        config['case_sensitive'] = temp_cs
    print(f"Finish: load dataset in {time.time()-start}")

    # apply filter:
    if 'text' in user_info.column_names and config['user_text_filter'] != "":
        if config['user_text_filter'] not in ["item_sentence_SBERT"]:
            user_info = filter_user_profile(config, user_info)

    # tokenize when needed:
    return_padding_token = None
    padding_token = None
    if pretrained_model is not None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
        padding_token = tokenizer.pad_token_id
        return_padding_token = tokenizer.pad_token_id
        if 'text' in user_info.column_names:
            user_info = user_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_user'] if "max_num_chunks_user" in config else None
                                                 })
            user_info = user_info.remove_columns(['text'])
        if 'text' in item_info.column_names:
            item_info = item_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_item'] if "max_num_chunks_item" in config else None
                                                 })
            item_info = item_info.remove_columns(['text'])

    user_used_items = None
    if 'random' in [config['training_neg_sampling_strategy'], config['validation_neg_sampling_strategy'],
                    config['test_neg_sampling_strategy']] or "genres" in [config['training_neg_sampling_strategy'], config['validation_neg_sampling_strategy'],
                    config['test_neg_sampling_strategy']]:
        start = time.time()
        print("Start: get user used items...")
        user_used_items = get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit)
        print(f"Finish: get user used items in {time.time()-start}")

    # previously we loaded the "textual profiles" when training into each batch...
    # now, we are going to create representations at initialization... and won't need that anymore.
    # todo: change1:
    # we are doing the check inside collatefns with padding token is None or not
    if config['text_in_batch'] is False:  # for now: means that we do pre calculation
        padding_token = None  # this causes the collate functions to

    train_collate_fn = None
    valid_collate_fn = None
    test_collate_fn = None
    if config['training_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and train collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        train_collate_fn = CollateNegSamplesRandomOpt(config['training_neg_samples'], cur_used_items, user_info,
                                                      item_info, padding_token=padding_token)
        print(f"Finish: used_item copy and train collate_fn initialize {time.time() - start}")
    elif config['training_neg_sampling_strategy'] == "":
        train_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token)
    elif config['training_neg_sampling_strategy'] == "genres":
        print("Start: used_item copy and train collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        train_collate_fn = CollateNegSamplesGenresOpt(config['training_neg_samples'], cur_used_items, user_info,
                                                      item_info, padding_token=padding_token)
        print(f"Finish: used_item copy and train collate_fn initialize {time.time() - start}")

    if config['validation_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and validation collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        for user_id, u_items in user_used_items['validation'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        print(f"Mid: used_item copy in {time.time() - start}")
        valid_collate_fn = CollateNegSamplesRandomOpt(config['validation_neg_samples'], cur_used_items,
                                                      padding_token=padding_token)
        print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")
    elif config['validation_neg_sampling_strategy'].startswith("f:"):
        valid_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token)
        # start = time.time()
        # print("Start: load negative samples and validation collate_fn initialize...")
        # negs = pd.read_csv(join(config['dataset_path'], config['validation_neg_sampling_strategy'][2:] + ".csv"))
        # negs = negs.merge(user_info.to_pandas()[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        # negs = negs.merge(item_info.to_pandas()[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        # negs = negs.drop(columns=["user_id", "item_id"])
        # print(f"Mid: negative samples loaded in {time.time() - start}")
        # valid_collate_fn = CollateNegSamplesFixed(negs, user_info, item_info, tokenizer.pad_token_id)
        # print(f"Finish: load negative samples and validation collate_fn initialize in {time.time() - start}")

    if config['test_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and test collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        for user_id, u_items in user_used_items['validation'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        for user_id, u_items in user_used_items['test'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        print(f"Mid: used_item copy in {time.time() - start}")
        test_collate_fn = CollateNegSamplesRandomOpt(config['test_neg_samples'], cur_used_items,
                                                     padding_token=padding_token)
        print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")
    elif config['test_neg_sampling_strategy'].startswith("f:"):
        test_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token)
        # start = time.time()
        # print("Start: load negative samples and test collate_fn initialize...")
        # negs = pd.read_csv(join(config['dataset_path'], config['test_neg_sampling_strategy'][2:] + ".csv"))
        # negs = negs.merge(user_info.to_pandas()[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        # negs = negs.merge(item_info.to_pandas()[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        # negs = negs.drop(columns=["user_id", "item_id"])
        # print(f"Mid: negative samples loaded in {time.time() - start}")
        # test_collate_fn = CollateNegSamplesFixed(negs, user_info, item_info, tokenizer.pad_token_id)
        # print(f"Finish: load negative samples and test collate_fn initialize in {time.time() - start}")



    # here goes the dataloaders from dataset objects returned above
    #### TODO tokenizer felan ina???

    # sampling negs: for training it is only train items,  for validation: train+valid, for test: train+valid+test

    train_dataloader = DataLoader(datasets['train'],
                                  batch_size=config['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=train_collate_fn,
                                  num_workers=config['dataloader_num_workers']
                                  ### sampler? how negative sampling is implemented into this? or should we do it outside?
                                  ### when creating the datasets?
                                  )
    validation_dataloader = DataLoader(datasets['validation'],
                                       batch_size=config['eval_batch_size'],
                                       collate_fn=valid_collate_fn,
                                       num_workers=config['dataloader_num_workers'])
    test_dataloader = DataLoader(datasets['test'],
                                 batch_size=config['eval_batch_size'],
                                 collate_fn=test_collate_fn,
                                 num_workers=config['dataloader_num_workers'])

    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, config['relevance_level'], return_padding_token


# class CollateNegSamples(object):
#     def __init__(self, strategy, num_neg_samples, used_items):
#         self.strategy = strategy
#         self.num_neg_samples = num_neg_samples
#         self.used_items = used_items
#         all_items = []
#         for items in self.used_items.values():
#             all_items.extend(items)
#         self.all_items = set(all_items)
#
#     def __call__(self, batch):
#         # TODO maybe change the sampling to like the fixed one... counter...
#         batch_df = pd.DataFrame(batch)
#         user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
#         samples = []
#         for user_id in user_counter.keys():
#             num_pos = user_counter[user_id]
#             num_user_neg_samples = num_pos * self.num_neg_samples
#             potential_items = self.all_items - self.used_items[user_id]
#             if len(potential_items) < num_user_neg_samples:
#                 print(f"WARNING: as there were not enough potential items to sample for user {user_id} with "
#                       f"{num_pos} positives needing {num_user_neg_samples} negs,"
#                       f"we reduced the number of user negative samples to potential items {len(potential_items)}"
#                       f"HOWEVER, bear in mind that this is problematic, since the negatives here are the positives in valid and test!")
#                 num_user_neg_samples = len(potential_items)
#             if self.strategy == 'random':
#                 for sampled_item in random.sample(potential_items, num_user_neg_samples):
#                     samples.append({'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item})
#         temp = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index()
#         ret = {}
#         for col in temp.columns:
#             ret[col] = torch.tensor(temp[col])
#         return ret

class CollateOriginalDataPad(object):
    def __init__(self, user_info, item_info, padding_token=None):
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        if self.padding_token is not None:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']]\
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])

            # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
            ret = {}
            for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids", "item_chunks_attention_mask"]:
                # instances = [pad_sequence([torch.tensor(t) for t in instance], padding_value=self.padding_token) for
                #              instance in temp[col]]
                instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
            for col in temp.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateRepresentationBuilder(object):
    def __init__(self, padding_token):
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        ret = {}
        for col in ["chunks_input_ids", "chunks_attention_mask"]:
            instances = [torch.tensor([list(t) for t in instance]) for instance in batch_df[col]]
            ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
        for col in batch_df.columns:
            if col in ret:
                continue
            if col in ["user_id", "item_id"]:
                continue
            ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateNegSamplesRandomOpt(object):
    def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in user_counter.keys():
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
            user_samples = set()
            try_cnt = -1
            num_user_neg_samples = max_num_user_neg_samples
            while True:
                if try_cnt == 100:
                    print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
                          f"{user_id}. We instead have {len(user_samples)} samples.")
                    break
                current_samples = set(random.sample(self.all_items, num_user_neg_samples))
                current_samples -= user_samples
                cur_used_samples = self.used_items[user_id].intersection(current_samples)
                current_samples = current_samples - cur_used_samples
                user_samples = user_samples.union(current_samples)
                num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                if len(user_samples) < max_num_user_neg_samples:
                    # to make the process faster
                    if num_user_neg_samples < len(user_samples):
                        num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                    try_cnt += 1
                else:
                    if len(user_samples) > max_num_user_neg_samples:
                        user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                    break
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in user_samples])
        batch_df = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index().drop(columns=['index'])

        # todo make this somehow that each of them could have text and better code
        if self.padding_token is not None:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])

            # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
            ret = {}
            for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
                        "item_chunks_attention_mask"]:
                # instances = [pad_sequence([torch.tensor(t) for t in instance], padding_value=self.padding_token) for
                #              instance in temp[col]]
                instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
            for col in temp.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateNegSamplesGenresOpt(object):
    def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        all_items = []  #?? it is set to not sample the positive valid/test items, but should it?
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.genres_field = 'category' if 'category' in self.item_info.columns else 'genres'
        self.genres_item = defaultdict(set)
        for item, genres in zip(self.item_info[INTERNAL_ITEM_ID_FIELD], self.item_info[self.genres_field]):
            for g in [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]:
                self.genres_item[g].add(item)

        self.padding_token = padding_token


    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        user_samples = defaultdict(set)
        batch_df = batch_df.merge(self.item_info[[self.genres_field, INTERNAL_ITEM_ID_FIELD]], on=INTERNAL_ITEM_ID_FIELD)
        for user_id, item_id, genres in zip(batch_df[INTERNAL_USER_ID_FIELD], batch_df[INTERNAL_ITEM_ID_FIELD], batch_df[self.genres_field]):
            genres = [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]
            candids = []
            for g in genres:
                candids.extend(list((self.genres_item[g]-set(self.used_items[user_id]))-user_samples[user_id]))
            candids = Counter(candids)
            # here I choose negative samples randomly, as each item has unique set of genre, however since items
            # have more than one genre, from which we may have in our item genres, we do it as such:
            # for eval negatives I do the sampling differently, as I have all pos items and their genres, there
            # is an additional count there.
            sampled_item_ids = np.random.choice(list(candids.keys()), min(len(candids), self.num_neg_samples),
                                                p=[c/sum(candids.values()) for c in candids.values()], replace=False)
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in sampled_item_ids])
            user_samples[user_id].update(set(sampled_item_ids))
        batch_df = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index().drop(columns=['index'])

        # todo make this somehow that each of them could have text and better code
        if self.padding_token is not None:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])

            # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
            ret = {}
            for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
                        "item_chunks_attention_mask"]:
                # instances = [pad_sequence([torch.tensor(t) for t in instance], padding_value=self.padding_token) for
                #              instance in temp[col]]
                instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
            for col in temp.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        else:
            ret = {}
            for col in batch[0].keys():
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret

# class CollateNegSamplesFixed(object):
#     def __init__(self, samples, user_info=None, item_info=None, padding_token=None):
#         self.samples_grouped = samples.groupby(by=[INTERNAL_USER_ID_FIELD])
#         self.user_info = user_info.to_pandas()
#         self.item_info = item_info.to_pandas()
#         self.padding_token = padding_token
#
#     def __call__(self, batch):
#         batch_df = pd.DataFrame(batch)
#         data = [batch_df]
#         for user_id in set(batch_df[INTERNAL_USER_ID_FIELD]):
#             data.append(self.samples_grouped.get_group(user_id))
#         batch_df = pd.concat(data).reset_index().drop(columns=['index'])
#         # user:
#         temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
#             .reset_index().drop(columns=['index'])
#         temp_user = pd.concat([batch_df, temp_user], axis=1)
#         temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
#                                               "chunks_attention_mask": "user_chunks_attention_mask"})
#         # item:
#         temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
#             .reset_index().drop(columns=['index'])
#         temp_item = pd.concat([batch_df, temp_item], axis=1)
#         temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
#                                               "chunks_attention_mask": "item_chunks_attention_mask"})
#         temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])
#
#         # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
#         ret = {}
#         for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
#                     "item_chunks_attention_mask"]:
#             # instances = [pad_sequence([torch.tensor(t) for t in instance], padding_value=self.padding_token) for
#             #              instance in temp[col]]
#             instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
#             ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
#         for col in temp.columns:
#             if col in ret:
#                 continue
#             ret[col] = torch.tensor(temp[col]).unsqueeze(1)
#         return ret


def get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = {}
        for user_iid, item_iid in zip(datasets[split]['internal_user_id'], datasets[split]['internal_item_id']):
            if user_iid not in used_items[split]:
                used_items[split][user_iid] = set()
            used_items[split][user_iid].add(item_iid)

    for user, books in filtered_out_user_item_pairs_by_limit.items():
        used_items['train'][user] = used_items['train'][user].union(books)

    return used_items


def load_split_dataset(config):
    user_text_fields = config['user_text']
    item_text_fields = config['item_text']
    if config['text_in_batch'] is False:
        user_text_fields = []
        item_text_fields = []

    # read users and items, create internal ids for them to be used

    # remove_fields = user_info.columns
    # print(f"user fields: {remove_fields}")
    keep_fields = ["user_id"]
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in user_text_fields if "user." in field])
    keep_fields.extend([field[field.index("user.")+len("user."):] for field in item_text_fields if "user." in field])
    keep_fields = list(set(keep_fields))
    # remove_fields = list(set(remove_fields) - set(keep_fields))
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"), usecols=keep_fields, dtype=str)
    # user_info = user_info.drop(columns=remove_fields)
    user_info = user_info.sort_values("user_id").reset_index(drop=True) # this is crucial, as the precomputing is done with internal ids
    user_info[INTERNAL_USER_ID_FIELD] = np.arange(0, user_info.shape[0])
    user_info = user_info.fillna('')
    user_info = user_info.rename(
        columns={field[field.index("user.") + len("user."):]: field for field in user_text_fields if
                 "user." in field})


    # print(f"item fields: {item_info.columns}")
    keep_fields = ["item_id"]
    keep_fields.extend([field[field.index("item.")+len("item."):] for field in item_text_fields if "item." in field])
    keep_fields.extend([field[field.index("item.") + len("item."):] for field in user_text_fields if "item." in field])
    keep_fields = list(set(keep_fields))
    tie_breaker = None
    if 'review_tie_breaker' in config:
        if config['review_tie_breaker'].startswith("item."):
            tie_breaker = config['review_tie_breaker']
            tie_breaker = tie_breaker[tie_breaker.index("item.") + len("item."):]
            keep_fields.extend([tie_breaker])
    if config["training_neg_sampling_strategy"] == "genres":
        if config["name"] == "Amazon":
            keep_fields.append("category")
        elif config["name"] == "CGR":
            keep_fields.append("genres")
        else:
            raise NotImplementedError()
        # item_info = item_info  TODO: what was this here for?
    # remove_fields = item_info.columns
    # remove_fields = list(set(remove_fields) - set(keep_fields))
    # item_info = item_info.drop(columns=remove_fields)
    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"), usecols=keep_fields, low_memory=False, dtype=str)
    if tie_breaker is not None:
        if tie_breaker == "avg_rating":
            item_info[tie_breaker] = item_info[tie_breaker].astype(float)
        elif tie_breaker == "rank":
            raise NotImplementedError("rank type to int")
        item_info[tie_breaker] = item_info[tie_breaker].fillna(0)
    item_info = item_info.sort_values("item_id").reset_index(drop=True)  # this is crucial, as the precomputing is done with internal ids
    item_info[INTERNAL_ITEM_ID_FIELD] = np.arange(0, item_info.shape[0])
    item_info = item_info.fillna('')
    # TODO add wrong genre removal, i.e. "like"
    item_info = item_info.rename(
        columns={field[field.index("item.") + len("item."):]: field for field in item_text_fields if
                 "item." in field})
    # TODO maybe move these preprocessing to another step? of creating the dataset?
    if config["name"] == "Amazon":
        if 'item.category' in item_info.columns:
            item_info['item.category'] = item_info['item.category'].apply(
                lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " "))
        if 'item.description' in item_info.columns:
            item_info['item.description'] = item_info['item.description'].apply(
                lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " "))
        # TODO rank one from str with , to int or something for tie breaker!

    # read user-item interactions, map the user and item ids to the internal ones
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], "validation.csv"),
                "test": join(config['dataset_path'], "test.csv")}
    split_datasets = {}
    filtered_out_user_item_pairs_by_limit = defaultdict(set)
    for sp, file in sp_files.items():
        df = pd.read_csv(file, dtype=str)  # rating:float64
        # book limit:
        if sp == 'train' and config['limit_training_data'] != "":
            if config['limit_training_data'].startswith("max_book"):
                limited_user_books = json.load(open(join(config['dataset_path'], f"{config['limit_training_data']}.json"), 'r'))
            else:
                raise NotImplementedError(f"limit_training_data={config['limit_training_data']} not implemented")

            limited_user_item_ids = []
            for user, books in limited_user_books.items():
                limited_user_item_ids.extend([f"{user}-{b}" for b in books])

            df['user_item_ids'] = df['user_id'].map(str) + '-' + df['item_id'].map(str)
            temp = df[df['user_item_ids'].isin(limited_user_item_ids)]['user_item_ids']
            temp = set(df['user_item_ids']) - set(temp)
            user_info_temp = user_info.copy()
            user_info_temp = user_info_temp.set_index('user_id')
            item_info_temp = item_info.copy()
            item_info_temp = item_info_temp.set_index('item_id')
            for ui in temp:
                user = int(user_info_temp.loc[int(ui[:ui.index('-')])].internal_user_id)
                item = int(item_info_temp.loc[int(ui[ui.index('-') + 1:])].internal_item_id)
                # if user not in filtered_out_user_item_pairs_by_limit:
                #     filtered_out_user_item_pairs_by_limit[user] = set()
                filtered_out_user_item_pairs_by_limit[user].add(item)
            df = df[df['user_item_ids'].isin(limited_user_item_ids)]
            df = df.drop(columns=['user_item_ids'])

        if config['binary_interactions']:
            # if binary prediction (interaction): set label for all rated/unrated/highrated/lowrated to 1.
            # TODO alternatively you can only consider interactions which have high ratings... filter out the rest...
            df['label'] = np.ones(df.shape[0])
            # df = df.drop(columns=['rating']) # todo we are not removing this now, bcs maybe we need it next when choosing the reviews
            df['rating'] = df['rating'].fillna(-1)
            if config["name"] == "CGR":
                for k, v in goodreads_rating_mapping.items():
                    df['rating'] = df['rating'].replace(k, v)
            elif config["name"] == "Amazon":
                df['rating'] = df['rating'].astype(float).astype(int)
            else:
                raise NotImplementedError(f"dataset {config['name']} not implemented!")
        else:
            # if predicting rating: remove the not-rated entries and map rating text to int
            df = df[df['rating'].notna()].reset_index()
            if config["name"] == "CGR":
                for k, v in goodreads_rating_mapping.items():
                    df['rating'] = df['rating'].replace(k, v)
            elif config["name"] == "Amazon":
                df['rating'] = df['rating'].astype(float).astype(int)
            else:
                raise NotImplementedError(f"dataset {config['name']} not implemented!")
            df['label'] = df['rating']
            # df = df.rename(columns={"rating": "label"})  # todo we are not removing this now, bcs maybe we need it next

        # replace user_id with internal_user_id (same for item_id)
        df = df.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        df = df.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        df = df.drop(columns=["user_id", "item_id"])

        df = df.rename(
            columns={field[field.index("interaction.") + len("interaction."):]: field for field in user_text_fields if
                     "interaction." in field})
        df = df.rename(
            columns={field[field.index("interaction.") + len("interaction."):]: field for field in item_text_fields if
                     "interaction." in field})

        # TODO preprocessing? removing html tags? ....
        for field in user_text_fields:
            if "interaction." in field:
                df[field] = df[field].fillna('')

        # concat and move the user/item text fields to user and item info:
        if "user_review_choice" not in config:
            sort_reviews = "rating_sorted"
        else:
            sort_reviews = config['user_review_choice']
        # text profile:
        if sp == 'train':
            ## USER:
            # This code works for user text fields from interaction and item file
            user_item_text_fields = [field for field in user_text_fields if "item." in field]
            user_inter_text_fields = [field for field in user_text_fields if "interaction." in field]
            user_item_inter_text_fields = user_item_text_fields.copy()
            user_item_inter_text_fields.extend(user_inter_text_fields)

            if len(user_item_inter_text_fields) > 0:
                user_item_merge_fields = [INTERNAL_ITEM_ID_FIELD]
                user_item_merge_fields.extend(user_item_text_fields)
                if tie_breaker == 'avg_rating':
                    user_item_merge_fields.append(tie_breaker)

                user_inter_merge_fields = [INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'rating']
                user_inter_merge_fields.extend(user_inter_text_fields)

                temp = df[user_inter_merge_fields].\
                    merge(item_info[user_item_merge_fields], on=INTERNAL_ITEM_ID_FIELD)
                if sort_reviews.startswith("pos_rating_sorted_"):
                    pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
                    temp = temp[temp['rating'] >= pos_threshold]
                # before sorting them based on rating, etc., let's append each row's field together (e.g. title. genres. review.)
                temp['text'] = temp[user_item_inter_text_fields].agg('. '.join, axis=1)

                if config['user_text_filter'] in ["item_sentence_SBERT"]:
                    # first we sort the items based on the ratings, tie-breaker
                    if tie_breaker is None:
                        temp = temp.sort_values(['rating'], ascending=[False])
                    else:
                        temp = temp.sort_values(['rating', tie_breaker], ascending=[False, False])

                    # sentencize the user text (r, tgr, ...)
                    sent_splitter = SentenceSplitter(language='en')
                    temp['sentences_text'] = temp.apply(lambda row: sentencize(row['text'], sent_splitter,
                                                                               config['case_sensitive'],
                                                                               config['normalize_negation']), axis=1)
                    # temp = temp.map(sentencize_function, fn_kwargs={
                    #     'case_sensitive': config['case_sensitive'],  # TODO set this to true?
                    #     'normalize_negation': config['normalize_negation'],
                    #     'sentencizer': sent_splitter
                    # }, batched=True)
                    temp = temp.drop(columns=['text'])
                    temp = temp.drop(columns=user_text_fields)

                    # load SBERT
                    sbert = SentenceTransformer("all-mpnet-base-v2")  # TODO hard coded
                    # "all-MiniLM-L12-v2"
                    # "all-MiniLM-L6-v2"
                    print("sentence transformer loaded!")

                    user_texts = []
                    for user_idx in list(user_info.index):
                        user = user_info.loc[user_idx][INTERNAL_USER_ID_FIELD]
                        user_items = []
                        user_item_temp = temp[temp[INTERNAL_USER_ID_FIELD] == user]
                        for item_id, sents in zip(user_item_temp[INTERNAL_ITEM_ID_FIELD], user_item_temp['sentences_text']):
                            if len(sents) == 0:
                                continue
                            item = item_info.loc[item_id]
                            if item_id != item[INTERNAL_ITEM_ID_FIELD]:
                                raise ValueError("item id and index does not match!")
                            item_text = '. '.join(list(item[item_text_fields]))
                            scores = util.dot_score(sbert.encode(item_text), sbert.encode(sents))
                            user_items.append([sent for score, sent in sorted(zip(scores[0], sents), reverse=True)])
                        # print(len(user_items))
                        user_text = []
                        cnts = {i: 0 for i in range(len(user_items))}
                        while True:
                            remaining = False
                            for i in range(len(user_items)):
                                if cnts[i] == len(user_items[i]):
                                    continue
                                remaining = True
                                user_text.append(user_items[i][cnts[i]])
                                cnts[i] += 1
                            if not remaining:
                                break
                        user_texts.append(' '.join(user_text))
                    user_info['text'] = user_texts
                    print(f"user text matching with item done!")
                else:
                    if sort_reviews == "nothing":
                        temp = temp.groupby(INTERNAL_USER_ID_FIELD)['text'].apply('. '.join).reset_index()
                    else:
                        if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):
                            # joined the text of user books(title, genre, review):  TODO before here need to do the thing for sorting each of these, and also consider cutting it.
                            if tie_breaker is None:
                                temp = temp.sort_values('rating', ascending=False).groupby(
                                    INTERNAL_USER_ID_FIELD)['text'].apply('. '.join).reset_index()
                            elif tie_breaker == "avg_rating":
                                temp = temp.sort_values(['rating', tie_breaker], ascending=[False, False]).groupby(
                                    INTERNAL_USER_ID_FIELD)['text'].apply('. '.join).reset_index()
                            else:
                                raise ValueError("Not implemented!")
                        else:
                            raise ValueError("Not implemented!")

                    user_info = user_info.merge(temp, "left", on=INTERNAL_USER_ID_FIELD)
                    user_info['text'] = user_info['text'].fillna('')

            ## ITEM:
            # This code works for item text fields from interaction and user file
            item_user_text_fields = [field for field in item_text_fields if "user." in field]
            item_inter_text_fields = [field for field in item_text_fields if "interaction." in field]
            item_user_inter_text_fields = item_user_text_fields.copy()
            item_user_inter_text_fields.extend(item_inter_text_fields)

            if len(item_user_inter_text_fields) > 0:
                item_user_merge_fields = [INTERNAL_USER_ID_FIELD]
                item_user_merge_fields.extend(item_user_text_fields)

                item_inter_merge_fields = [INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'rating']
                item_inter_merge_fields.extend(item_inter_text_fields)

                temp = df[item_inter_merge_fields]. \
                    merge(user_info[item_user_merge_fields], on=INTERNAL_USER_ID_FIELD)
                if sort_reviews.startswith("pos_rating_sorted_"):  # Todo sort_review field in config new?
                    pos_threshold = int(sort_reviews[sort_reviews.rindex("_") + 1:])
                    temp = temp[temp['rating'] >= pos_threshold]
                # before sorting them based on rating, etc., let's append each row's field together
                temp['text'] = temp[item_user_inter_text_fields].agg('. '.join, axis=1)

                if sort_reviews == "nothing":
                    temp = temp.groupby(INTERNAL_ITEM_ID_FIELD)['text'].apply('. '.join).reset_index()
                else:
                    if sort_reviews == "rating_sorted" or sort_reviews.startswith("pos_rating_sorted_"):
                        temp = temp.sort_values('rating', ascending=False).groupby(
                            INTERNAL_ITEM_ID_FIELD)['text'].apply('. '.join).reset_index()
                    else:
                        raise ValueError("Not implemented!")

                item_info = item_info.merge(temp, "left", on=INTERNAL_ITEM_ID_FIELD)
                item_info['text'] = item_info['text'].fillna('')

        # remove the rest
        remove_fields = df.columns
        print(f"interaction fields: {remove_fields}")
        keep_fields = ["label", INTERNAL_ITEM_ID_FIELD, INTERNAL_USER_ID_FIELD]
        remove_fields = list(set(remove_fields) - set(keep_fields))
        df = df.drop(columns=remove_fields)
        split_datasets[sp] = df

    # TODO the SBERT match with item desc should also be applied here?? guess not
    # after moving text fields to user/item info, now concatenate them all and create a single 'text' field:
    user_remaining_text_fields = [field for field in user_text_fields if field.startswith("user.")]
    if 'text' in user_info.columns:
        user_remaining_text_fields.append('text')
    if len(user_remaining_text_fields) > 0:
        user_info['text'] = user_info[user_remaining_text_fields].agg('. '.join, axis=1)
        if not config['case_sensitive']:
            user_info['text'] = user_info['text'].apply(str.lower)
        if config['normalize_negation']:
            user_info['text'] = user_info['text'].replace("n\'t", " not", regex=True)
        user_info = user_info.drop(columns=[field for field in user_text_fields if field.startswith("user.")])

    item_remaining_text_fields = [field for field in item_text_fields if field.startswith("item.")]
    if 'text' in item_info.columns:
        item_remaining_text_fields.append('text')
    if len(item_remaining_text_fields) > 0:
        item_info['text'] = item_info[item_remaining_text_fields].agg('. '.join, axis=1)
        if not config['case_sensitive']:
            item_info['text'] = item_info['text'].apply(str.lower)
        if config['normalize_negation']:
            item_info['text'] = item_info['text'].replace("n\'t", " not", regex=True)
        item_info = item_info.drop(columns=[field for field in item_text_fields if field.startswith("item.")])

    # loading negative samples for eval sets: I used to load them in a collatefn, but, because batch=101 does not work for evaluation for BERT-based models
    # I would load them here.
    if config['validation_neg_sampling_strategy'].startswith("f:"):
        negs = pd.read_csv(join(config['dataset_path'], config['validation_neg_sampling_strategy'][2:]+".csv"), dtype=str)
        negs['label'] = negs['label'].astype(int)
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        split_datasets['validation'] = pd.concat([split_datasets['validation'], negs])
        split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    if config['test_neg_sampling_strategy'].startswith("f:"):
        negs = pd.read_csv(join(config['dataset_path'], config['test_neg_sampling_strategy'][2:] + ".csv"), dtype=str)
        negs['label'] = negs['label'].astype(int)
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        split_datasets['test'] = pd.concat([split_datasets['test'], negs])
        split_datasets['test'] = split_datasets['test'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    for split in split_datasets.keys():
        split_datasets[split] = Dataset.from_pandas(split_datasets[split], preserve_index=False)

    return DatasetDict(split_datasets), Dataset.from_pandas(user_info, preserve_index=False), \
           Dataset.from_pandas(item_info, preserve_index=False), filtered_out_user_item_pairs_by_limit



# load_data({"dataset_path": "/home/ghazaleh/workspace/SBR/data/GR_read_5-folds/toy_dataset/",
#             "binary_interactions": True, "name": "CGR", "batch_size": 8})
### returing all chunks:  return_overflowing_tokens
# def tokenize_and_split(examples):
#     return tokenizer(
#         examples["review"],
#         truncation=True,
#         max_length=128,
#         return_overflowing_tokens=True,
#     )
# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
### but this gives error because some fields of the dataset are longer than others (input id)
### we can remove other fields and just have our new dataset:
# tokenized_dataset = drug_dataset.map(
#     tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
# )
### or we can extend the old dataset features to match the new feature size
# def tokenize_and_split(examples):
#     result = tokenizer(
#         examples["review"],
#         truncation=True,
#         max_length=128,
#         return_overflowing_tokens=True,
#     )
#     # Extract mapping between new and old indices
#     sample_map = result.pop("overflow_to_sample_mapping")
#     for key, values in examples.items():
#         result[key] = [values[i] for i in sample_map]
#     return result
# tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
