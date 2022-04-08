import csv
import random
import time
from collections import Counter
from os.path import join

import pandas
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

goodreads_rating_mapping = {
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def load_data(config):
    start = time.time()
    print("Start: load dataset...")
    if config["name"] == "CGR":
        datasets, user_info, item_info = load_crawled_goodreads_dataset(config)
    else:
        raise ValueError(f"dataset {config['name']} not implemented!")
    print(f"Finish: load dataset in {time.time()-start}")

    user_used_items = None
    if 'random' in [config['training_neg_sampling_strategy'], config['validation_neg_sampling_strategy'],
                    config['test_neg_sampling_strategy']]:
        start = time.time()
        print("Start: get user used items...")
        user_used_items = get_user_used_items(datasets)
        print(f"Finish: get user used items in {time.time()-start}")

    train_collate_fn = None
    valid_collate_fn = None
    test_collate_fn = None
    if config['training_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and train collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        print(f"Mid: used_item copy in {time.time() - start}")
        train_collate_fn = CollateNegSamplesRandomOpt(config['training_neg_samples'], cur_used_items)
        print(f"Finish: used_item copy and train collate_fn initialize {time.time() - start}")

    if config['validation_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and validation collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        for user_id, u_items in user_used_items['validation'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        print(f"Mid: used_item copy in {time.time() - start}")
        valid_collate_fn = CollateNegSamplesRandomOpt(config['validation_neg_samples'], cur_used_items)
        print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")
    elif config['validation_neg_sampling_strategy'].startswith("f:"):
        start = time.time()
        print("Start: load negative samples and validation collate_fn initialize...")
        negs = pd.read_csv(join(config['dataset_path'], config['validation_neg_sampling_strategy'][2:] + ".csv"))
        negs = negs.merge(user_info, "left", on="user_id")
        negs = negs.merge(item_info, "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        print(f"Mid: negative samples loaded in {time.time() - start}")
        valid_collate_fn = CollateNegSamplesFixed(negs)
        print(f"Finish: load negative samples and validation collate_fn initialize in {time.time() - start}")

    if config['test_neg_sampling_strategy'] == "random":
        start = time.time()
        print("Start: used_item copy and test collate_fn initialize...")
        cur_used_items = user_used_items['train'].copy()
        for user_id, u_items in user_used_items['validation'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        for user_id, u_items in user_used_items['test'].items():
            cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
        print(f"Mid: used_item copy in {time.time() - start}")
        test_collate_fn = CollateNegSamplesRandomOpt(config['test_neg_samples'], cur_used_items)
        print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")
    elif config['test_neg_sampling_strategy'].startswith("f:"):
        start = time.time()
        print("Start: load negative samples and test collate_fn initialize...")
        negs = pd.read_csv(join(config['dataset_path'], config['test_neg_sampling_strategy'][2:] + ".csv"))
        negs = negs.merge(user_info, "left", on="user_id")
        negs = negs.merge(item_info, "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        print(f"Mid: negative samples loaded in {time.time() - start}")
        test_collate_fn = CollateNegSamplesFixed(negs)
        print(f"Finish: load negative samples and test collate_fn initialize in {time.time() - start}")

        # cur_used_items.update(user_used_items['test'])
        # test_collate_fn = CollateNegSamples(config['evaluation_neg_sampling_strategy'],
        #                                      config['evaluation_neg_samples'], cur_used_items)


    # here goes the dataloaders from dataset objects returned above
    #### TODO tokenizer felan ina???

    # sampling negs: for training it is only train items,  for validation: train+valid, for test: train+valid+test

    train_dataloader = DataLoader(datasets['train'],
                                  batch_size=config['train_batch_size'],
                                  shuffle=True,
                                  collate_fn=train_collate_fn
                                  ### sampler? how negative sampling is implemented into this? or should we do it outside?
                                  ### when creating the datasets?
                                  )
    validation_dataloader = DataLoader(datasets['validation'],
                                       batch_size=config['eval_batch_size'],
                                       collate_fn=valid_collate_fn)
    test_dataloader = DataLoader(datasets['test'],
                                 batch_size=config['eval_batch_size'],
                                 collate_fn=test_collate_fn)

    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, config['relevance_level']


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
#         temp = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index() ## TODO what if there are more fields?
#         ret = {}
#         for col in temp.columns:
#             ret[col] = torch.tensor(temp[col])
#         return ret


class CollateNegSamplesRandomOpt(object):
    def __init__(self, num_neg_samples, used_items):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))

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
                if len(user_samples) < max_num_user_neg_samples:
                    current_samples = current_samples - cur_used_samples
                    user_samples = user_samples.union(current_samples)
                    num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
                    # to make the process faster
                    if num_user_neg_samples < len(user_samples):
                        num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
                    try_cnt += 1
                else:
                    user_samples = user_samples.union(current_samples)
                    if len(user_samples) > max_num_user_neg_samples:
                        user_samples = set(list(user_samples)[:max_num_user_neg_samples])
                    break
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in user_samples])
        temp = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index()  ## TODO what if there are more fields?
        ret = {}
        for col in temp.columns:
            ret[col] = torch.tensor(temp[col])
        return ret


class CollateNegSamplesFixed(object):
    def __init__(self, samples):
        users = set(samples[INTERNAL_USER_ID_FIELD])
        self.samples = {}
        for user_id in users:
            self.samples[user_id] = samples[samples[INTERNAL_USER_ID_FIELD] == user_id].reset_index()

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        data = [batch_df]
        for user_id in set(batch_df[INTERNAL_USER_ID_FIELD]):
            data.append(self.samples[user_id])
        temp = pd.concat(data).reset_index() ## TODO what if there are more fields?
        ret = {}
        for col in temp.columns:
            ret[col] = torch.tensor(temp[col])
        return ret


def get_user_used_items(datasets):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = {}
        for user_iid, item_iid in zip(datasets[split]['internal_user_id'], datasets[split]['internal_item_id']):
            if user_iid not in used_items[split]:
                used_items[split][user_iid] = set()
            used_items[split][user_iid].add(item_iid)

    return used_items


def load_crawled_goodreads_dataset(config):
    # read users and items, create internal ids for them to be used
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"))
    remove_fields = user_info.columns
    print(f"user fields: {remove_fields}")
    keep_fields = ["user_id"]
    keep_fields.extend(config['user_text'])
    remove_fields = list(set(remove_fields) - set(keep_fields))
    user_info = user_info.drop(columns=remove_fields)
    user_info = user_info.sort_values("user_id")
    user_info[INTERNAL_USER_ID_FIELD] = np.arange(0, user_info.shape[0])

    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"))
    remove_fields = item_info.columns
    print(f"item fields: {remove_fields}")
    keep_fields = ["item_id"]
    keep_fields.extend(config['item_text'])
    remove_fields = list(set(remove_fields) - set(keep_fields))
    item_info = item_info.drop(columns=remove_fields)
    item_info = item_info.sort_values("item_id")
    item_info[INTERNAL_ITEM_ID_FIELD] = np.arange(0, item_info.shape[0])

    # read user-item interactions, map the user and item ids to the internal ones
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], "validation.csv"),
                "test": join(config['dataset_path'], "test.csv")}
    split_datasets = {}
    for sp, file in sp_files.items():
        df = pd.read_csv(file)
        df['review'] = df['review'].fillna('')

        if config['binary_interactions']:
            # if binary prediction (interaction): set label for all rated/unrated/highrated/lowrated to 1.
            # TODO alternatively you can only consider interactions which have high ratings... filter out the rest...
            df['label'] = np.ones(df.shape[0])
            df = df.drop(columns=['rating'])
        else:
            # if predicting rating: remove the not-rated entries and map rating text to int
            df = df[df['rating'].notna()].reset_index()
            for k, v in goodreads_rating_mapping.items():
                df['rating'] = df['rating'].replace(k, v)
            df = df.rename(columns={"rating": "label"})

        remove_fields = df.columns
        print(f"interaction fields: {remove_fields}")
        keep_fields = ["label", "user_id", "item_id"]
        keep_fields.extend(config['user_text'])
        keep_fields.extend(config['item_text'])
        remove_fields = list(set(remove_fields) - set(keep_fields))
        df = df.drop(columns=remove_fields)
        df = df.merge(user_info, "left", on="user_id")
        df = df.merge(item_info, "left", on="item_id")
        df = df.drop(columns=["user_id", "item_id"])
        split_datasets[sp] = df

    # loading negative samples for eval sets:
    # if config['validation_neg_sampling_strategy'].startswith("f:"):
    #     negs = pd.read_csv(join(config['dataset_path'], config['validation_neg_sampling_strategy'][2:]+".csv"))
    #     negs = negs.merge(user_info, "left", on="user_id")
    #     negs = negs.merge(item_info, "left", on="item_id")
    #     negs = negs.drop(columns=["user_id", "item_id"])
    #     split_datasets['validation'] = pd.concat([split_datasets['validation'], negs])
    #     split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD)
    #
    # if config['test_neg_sampling_strategy'].startswith("f:"):
    #     negs = pd.read_csv(join(config['dataset_path'], config['test_neg_sampling_strategy'][2:] + ".csv"))
    #     negs = negs.merge(user_info, "left", on="user_id")
    #     negs = negs.merge(item_info, "left", on="item_id")
    #     negs = negs.drop(columns=["user_id", "item_id"])
    #     split_datasets['test'] = pd.concat([split_datasets['test'], negs])
    #     split_datasets['test'] = split_datasets['test'].sort_values(INTERNAL_USER_ID_FIELD)

    for split in split_datasets.keys():
        split_datasets[split] = Dataset.from_pandas(split_datasets[split], preserve_index=False)

    # TODO preprocessing to text? lowercaser? or not
    return DatasetDict(split_datasets), user_info, item_info  # user item info are pandas for now


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
