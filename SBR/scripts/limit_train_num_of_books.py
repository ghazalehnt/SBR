import argparse
import csv
import json
import random
from os.path import join

import numpy as np


def read_data(dataset_path):
    ret = {}
    sp_files = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
    for split, sp_file in sp_files.items():
        ret[split] = []
        with open(join(dataset_path, sp_file), 'r') as f:
            reader = csv.reader(f)
            h = next(reader)
            for line in reader:
                ret[split].append(line)

    user_info = []
    with open(join(dataset_path, f"users.csv"), 'r') as f:
        reader = csv.reader(f)
        user_header = next(reader)
        for line in reader:
            user_info.append(line)

    item_info = []
    with open(join(dataset_path, f"items.csv"), 'r') as f:
        reader = csv.reader(f)
        item_header = next(reader)
        for line in reader:
            item_info.append(line)

    return ret["train"], ret["validation"], ret["test"], h, user_info, user_header, item_info, item_header


def limit_train_books_uniformly(orig_train, header, max_books):
    new_train = {}

    USER_IDX = header.index("user_id")
    ITEM_IDX = header.index("item_id")
    user_books = {}
    for line in orig_train:
        if line[USER_IDX] not in user_books:
            user_books[line[USER_IDX]] = []
        user_books[line[USER_IDX]].append(line[ITEM_IDX])

    for user, books in user_books.items():
        new_train[user] = list(np.random.choice(books, min(max_books, len(books)), replace=False))

    return new_train


def main(data_dir, max_book, sample_strategy):
    train, valid, test, inter_header, all_users, user_header, all_items, item_header = read_data(data_dir)

    # user id -> [book ids,...]
    if sample_strategy == 'uniform':
        new_train = limit_train_books_uniformly(train, inter_header, max_book)
    else:
        raise NotImplementedError(f"{sample_strategy} not implemented")

    with open(join(data_dir, f"max_book_{max_book}_{sample_strategy}.json"), 'w') as f:
        json.dump(new_train, f)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', '-d', type=str, help='path to dataset')
    parser.add_argument('--max_book', '-m', type=int, help='number of max book samples')
    parser.add_argument('--strategy', '-s', type=str, help='strategy of how the books are sampled', default='uniform')
    args, _ = parser.parse_known_args()
    dataset_dir = args.dataset_folder
    book_limit = args.max_book
    strategy = args.strategy

    main(dataset_dir, book_limit, strategy)
    