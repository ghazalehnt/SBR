import argparse
import json
from os.path import join

import torch
import pandas as pd
import transformers
from datasets import Dataset

from SBR.utils.others import get_model


def tokenize_function(examples, tokenizer, field, max_length, padding):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        padding=padding  # we pad the chunks here, because it would be too complicated later due to the chunks themselves...
    )
    examples['input_ids'] = result['input_ids']
    examples['attention_mask'] = result['attention_mask']
    return examples


def main(exp_path, item_file):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open(join(exp_path, "config.json"), 'r'))

    items = pd.read_csv(item_file, dtype=str)
    items = items.fillna("")
    # items = items.sort_values("item_id").reset_index(drop=True)  # do we need?
    # items[INTERNAL_ITEM_ID_FIELD] = np.arange(0, items.shape[0])
    items = Dataset.from_pandas(items)
    tokenizer = transformers.AutoTokenizer.from_pretrained(config["model"]["pretrained_model"])
    padding_token = tokenizer.pad_token_id
    items = items.map(tokenize_function, batched=True,
                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                 "max_length": config["dataset"]["item_chunk_size"],
                                 "padding": False})
    items = items.remove_columns(['text'])

    model = get_model(config['model'], None, None, device, config['dataset'], test_only=True)
    reps, ex_item_ids = model.prec_representations_given_items(items, padding_token=padding_token)
    json.dump({i: v for i, v in zip(ex_item_ids, reps)}, open(join(exp_path, "all_item_prec_output.json"), 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', '-r', type=str, default=None, help='path to checkpoint and (model) config')
    parser.add_argument('--item_profile_file', default=None, help='path to the item profile file that we want to prec')
    args, _ = parser.parse_known_args()
    main(args.checkpoint_path, args.item_profile_file)

