import csv
from os.path import join

import pandas as pd
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import numpy as np

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

goodreads_rating_mapping = {
    None: None,  ## this means there was no rating
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}


def load_data(config):
    if config["name"] == "CGR":
        datasets, user_info, item_info = load_crawled_goodreads_dataset(config)
    else:
        raise ValueError(f"dataset {config['name']} not implemented!")

    # here goes the dataloaders from dataset objects returned above
    #### TODO tokenizer felan ina???
    train_dataloader = DataLoader(datasets["train"], batch_size=config["batch_size"],
                                  # shuffle=
                                  # collate_fn=
                                  ### sampler? how negative sampling is implemented into this? or should we do it outside?
                                  ### when creating the datasets?
                                  )
    validation_dataloader = DataLoader(datasets["validation"], batch_size=config["batch_size"])
    test_dataloader = DataLoader(datasets["test"], batch_size=config["batch_size"])

    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, config["relevance_level"]


def load_crawled_goodreads_dataset(config):
    # read users and items, create internal ids for them to be used
    user_info = load_dataset("csv", data_files=join(config['dataset_path'], "users.csv"))["train"]
    remove_fields = user_info.column_names
    print(f"user fields: {remove_fields}")
    keep_fields = ["user_id"]
    keep_fields.extend(config['user_text'])
    remove_fields = list(set(remove_fields) - set(keep_fields))
    user_info = user_info.remove_columns(remove_fields)
    user_info = user_info.sort("user_id")
    user_info = user_info.add_column(INTERNAL_USER_ID_FIELD, np.arange(0, user_info.num_rows))

    item_info = load_dataset("csv", data_files=join(config['dataset_path'], "items.csv"))["train"]
    remove_fields = item_info.column_names
    print(f"item fields: {remove_fields}")
    keep_fields = ["item_id"]
    keep_fields.extend(config['item_text'])
    remove_fields = list(set(remove_fields) - set(keep_fields))
    item_info = item_info.remove_columns(remove_fields)
    item_info = item_info.sort("item_id")
    item_info = item_info.add_column(INTERNAL_ITEM_ID_FIELD, np.arange(0, item_info.num_rows))

    # read user-item interactions, map the user and item ids to the internal ones
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], "validation.csv"),
                "test": join(config['dataset_path'], "test.csv")}

    split_datasets = load_dataset("csv", data_files=sp_files)
    split_datasets = split_datasets.map(lambda x: {'review': x['review'] if x['review'] is not None else ''})

    if config['binary_interactions']:
        # if binary prediction (interaction): set label for all rated/unrated/highrated/lowrated to 1.
        # TODO alternatively you can only consider interactions which have high ratings... filter out the rest...
        split_datasets = split_datasets.map(lambda x: {'label': 1})
    else:
        # if predicting rating: remove the not-rated entries and map rating text to int
        split_datasets = split_datasets.map(lambda x: {'rating': goodreads_rating_mapping[x['rating']]})
        split_datasets = split_datasets.filter(lambda x: x['rating'] is not None)
        split_datasets = split_datasets.rename_column("rating", "label")

    remove_fields = split_datasets.column_names["train"]
    print(f"interaction fields: {remove_fields}")
    keep_fields = ["label", "user_id", "item_id"]
    keep_fields.extend(config['user_text'])
    keep_fields.extend(config['item_text'])
    remove_fields = list(set(remove_fields) - set(keep_fields))
    split_datasets = split_datasets.remove_columns(remove_fields)
    for split in sp_files.keys():
        dataset_pd = split_datasets[split].data.to_pandas()
        dataset_pd = dataset_pd.merge(user_info.to_pandas(), "left", on="user_id")
        dataset_pd = dataset_pd.merge(item_info.to_pandas(), "left", on="item_id")
        split_datasets[split] = Dataset.from_pandas(dataset_pd, preserve_index=False)
    split_datasets = split_datasets.remove_columns(["user_id", "item_id"])
    # split_datasets.set_format("pt")
    # ## TODO format?


    # TODO preprocessing to text? lowercaser? or not
    return split_datasets, user_info, item_info


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
