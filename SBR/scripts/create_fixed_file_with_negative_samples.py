import argparse
import csv
import os
import random
from collections import Counter

from datasets import load_dataset


def load_data(dataset_path):
    sp_files = {"train": os.path.join(dataset_path, "train.csv"),
                "validation": os.path.join(dataset_path, "validation.csv"),
                "test": os.path.join(dataset_path, "test.csv")}
    split_datasets = load_dataset("csv", data_files=sp_files)
    return split_datasets


def get_user_used_items(datasets):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = {}
        for user_iid, item_iid in zip(datasets[split]['user_id'], datasets[split]['item_id']):
            if user_iid not in used_items[split]:
                used_items[split][user_iid] = set()
            used_items[split][user_iid].add(item_iid)

    return used_items

### BUG used item in the samples
def neg_sampling(data, used_items, strategy, num_neg_samples):
    all_items = []
    for items in used_items.values():
        all_items.extend(items)
    all_items = set(all_items)
    samples = []
    user_counter = Counter(data['user_id'])
    for user_id in user_counter.keys():
        num_pos = user_counter[user_id]
        num_user_neg_samples = num_pos * num_neg_samples
        potential_items = list(all_items - used_items[user_id])
        if len(potential_items) < num_user_neg_samples:
            print(f"WARNING: as there were not enough potential items to sample for user {user_id} with "
                  f"{num_pos} positives needing {num_user_neg_samples} negs,"
                  f"we reduced the number of user negative samples to potential items {len(potential_items)}"
                  f"HOWEVER, bear in mind that this is problematic, as the validation has 0s for 1s of test!")
            num_user_neg_samples = len(potential_items)
        if strategy == 'random':
            for sampled_item in random.sample(potential_items, num_user_neg_samples):
                samples.append([user_id, sampled_item, 0])
    return samples


def main(dataset_path, strategy, num_neg_samples):
    datasets = load_data(dataset_path)
    user_used_items = get_user_used_items(datasets)

    used_items = user_used_items['train']
    used_items.update(user_used_items['validation'])
    validation_samples = neg_sampling(datasets['validation'], used_items, strategy, num_neg_samples)
    with open(os.path.join(dataset_path, f'validation_neg_{strategy}_{num_neg_samples}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'label'])
        writer.writerows(validation_samples)

    used_items.update(user_used_items['test'])
    test_samples = neg_sampling(datasets['test'], used_items, strategy, num_neg_samples)
    with open(os.path.join(dataset_path, f'test_neg_{strategy}_{num_neg_samples}.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'item_id', 'label'])
        writer.writerows(test_samples)

if __name__ == "__main__":
    random.seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder', '-d', type=str, help='path to dataset')
    parser.add_argument('--ns', type=int, help='number of negative samples')
    args, _ = parser.parse_known_args()

    main(args.dataset_folder, "random", args.ns)
