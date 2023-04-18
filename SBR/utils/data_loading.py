import json
import pickle
import random
import time
from builtins import NotImplementedError
from collections import Counter, defaultdict, OrderedDict
from os.path import join, exists

import pandas as pd
import torch
import torchtext
import transformers
from datasets import Dataset, DatasetDict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
from torchtext.data import get_tokenizer
import gensim

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

goodreads_rating_mapping = {
    'did not like it': 1,
    'it was ok': 2,
    'liked it': 3,
    'really liked it': 4,
    'it was amazing': 5
}

def tokenize_function_torchtext(examples, tokenizer, field, vocab):
    result = [vocab(tokenizer(text)) for text in examples[field]]
    examples[f"tokenized_{field}"] = result
    return examples


def tokenize_function_many_chunks(examples, tokenizer, field, max_length, max_num_chunks, padding):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        padding=padding  # we pad the chunks here, because it would be too complicated later due to the chunks themselves...
    )

    sample_map = result.pop("overflow_to_sample_mapping")
    examples['chunks_input_ids'] = [[] for i in range(len(examples[field]))]
    examples['chunks_attention_mask'] = [[] for i in range(len(examples[field]))]
    for i, j in zip(sample_map, range(len(result['input_ids']))):
        if max_num_chunks is None or len(examples['chunks_input_ids'][i]) < max_num_chunks:
            examples['chunks_input_ids'][i].append(result['input_ids'][j])
            examples['chunks_attention_mask'][i].append(result['attention_mask'][j])
    return examples


def tokenize_function(examples, tokenizer, field, max_length, max_num_chunks, padding):
    result = tokenizer(
        examples[field],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        padding=padding  # we pad the chunks here, because it would be too complicated later due to the chunks themselves...
    )

    sample_map = result.pop("overflow_to_sample_mapping")
    examples['chunks_input_ids'] = [[] for i in range(len(examples[field]))]
    examples['chunks_attention_mask'] = [[] for i in range(len(examples[field]))]
    for i, j in zip(sample_map, range(len(result['input_ids']))):
        if max_num_chunks is None or len(examples['chunks_input_ids'][i]) < max_num_chunks:
            examples['chunks_input_ids'][i].append(result['input_ids'][j])
            examples['chunks_attention_mask'][i].append(result['attention_mask'][j])
    return examples


def load_data(config, pretrained_model=None, word_vector_model=None, for_precalc=False, exp_dir=None, joint=False):
    start = time.time()
    print("Start: load dataset...")
    datasets, user_info, item_info, filtered_out_user_item_pairs_by_limit = load_split_dataset(config, for_precalc)
    print(f"Finish: load dataset in {time.time()-start}")

    # tokenize when needed:
    return_padding_token = None
    padding_token = None
    if pretrained_model is not None and config["load_tokenized_text_in_batch"] is True:
        tokenizer = transformers.AutoTokenizer.from_pretrained(pretrained_model)
        padding_token = tokenizer.pad_token_id
        return_padding_token = tokenizer.pad_token_id
        if 'text' in user_info.column_names:
            user_info = user_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["user_chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_user'] if "max_num_chunks_user" in config else None,
                                                 "padding": False  # TODO should be fixed later for more chunks ...  to pad correctly
                                                 })
            user_info = user_info.remove_columns(['text'])
        if 'text' in item_info.column_names:
            item_info = item_info.map(tokenize_function, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text',
                                                 # this is used to know how big should the chunks be, because the model may have extra stuff to add to the chunks
                                                 "max_length": config["item_chunk_size"],
                                                 "max_num_chunks": config['max_num_chunks_item'] if "max_num_chunks_item" in config else None,
                                                 "padding": False # TODO should be fixed later for more chunks ...  to pad correctly
                                                 })
            item_info = item_info.remove_columns(['text'])
    elif word_vector_model is not None and config["load_tokenized_text_in_batch"] is True:
        tokenizer = get_tokenizer("basic_english")
        padding_token = 0
        if exists(join(exp_dir, "vocab.pth")):
            wv_vocab = torch.load(join(exp_dir, "vocab.pth"))
        else:
            word_embedding = gensim.models.KeyedVectors.load_word2vec_format(word_vector_model)
            vocab = word_embedding.index_to_key
            wv_vocab = torchtext.vocab.vocab(OrderedDict([(token, 1) for token in vocab]))
            wv_vocab.insert_token("<pad>", 0)
            wv_vocab.insert_token("<unk>", 1)
            wv_vocab.set_default_index(wv_vocab["<unk>"])
            torch.save(wv_vocab, join(exp_dir, "vocab.pth"))
        if 'text' in user_info.column_names:
            user_info = user_info.map(tokenize_function_torchtext, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text', "vocab": wv_vocab})
        if 'text' in item_info.column_names:
            item_info = item_info.map(tokenize_function_torchtext, batched=True,
                                      fn_kwargs={"tokenizer": tokenizer, "field": 'text', "vocab": wv_vocab})

    train_dataloader, validation_dataloader, test_dataloader = None, None, None
    if not for_precalc:
        start = time.time()
        print("Start: get user used items...")
        user_used_items = get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit)
        print(f"Finish: get user used items in {time.time() - start}")

        # when we need text for the training. we sort of check it if the passed padding_token is not none in collate_fns, so this is set here now:
        if config['load_tokenized_text_in_batch'] is False:
            padding_token = None  # this causes the collate functions to

        train_collate_fn = None
        valid_collate_fn = None
        test_collate_fn = None
        start = time.time()
        if config['training_neg_sampling_strategy'] == "random":
            print("Start: train collate_fn initialize...")
            cur_used_items = user_used_items['train'].copy()
            train_collate_fn = CollateNegSamplesRandomOpt(config['training_neg_samples'],
                                                          cur_used_items, user_info,
                                                          item_info, padding_token=padding_token, joint=joint)
            print(f"Finish: train collate_fn initialize {time.time() - start}")
        elif config['training_neg_sampling_strategy'].startswith("random_w_CF_dot_"):
            cur_used_items = user_used_items['train'].copy()
            label_weight_name = config['training_neg_sampling_strategy']
            oldmax = int(label_weight_name[label_weight_name.rindex("_") + 1:])
            oldmin = int(label_weight_name[
                         label_weight_name.index("w_CF_dot_") + len("w_CF_dot_"):label_weight_name.rindex("_")])
            train_collate_fn = CollateNegSamplesRandomCFWeighted(config['training_neg_samples'],
                                                                 cur_used_items,
                                                                 user_used_items['train'],
                                                                 config["cf_sim_checkpoint"],
                                                                 config["cf_internal_ids"],
                                                                 oldmax,
                                                                 oldmin,
                                                                 user_info,
                                                                 item_info, padding_token=padding_token, joint=joint)
            print(f"Finish: train collate_fn initialize {time.time() - start}")
        elif config['training_neg_sampling_strategy'] == "":
            train_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token)
        elif config['training_neg_sampling_strategy'] == "genres":
            print("Start: train collate_fn initialize...")
            cur_used_items = user_used_items['train'].copy()
            train_collate_fn = CollateNegSamplesGenresOpt(config['training_neg_sampling_strategy'],
                                                          config['training_neg_samples'], cur_used_items, user_info,
                                                          item_info, padding_token=padding_token, joint=joint)
            print(f"Finish: train collate_fn initialize {time.time() - start}")

        if config['validation_neg_sampling_strategy'] == "random":
            start = time.time()
            print("Start: used_item copy and validation collate_fn initialize...")
            cur_used_items = user_used_items['train'].copy()
            for user_id, u_items in user_used_items['validation'].items():
                cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
            valid_collate_fn = CollateNegSamplesRandomOpt(config['validation_neg_samples'], cur_used_items,
                                                          padding_token=padding_token, joint=joint)
            print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")
        elif config['validation_neg_sampling_strategy'].startswith("f:"):
            start = time.time()
            print("Start: used_item copy and validation collate_fn initialize...")
            valid_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token, joint=joint)
            print(f"Finish: used_item copy and validation collate_fn initialize {time.time() - start}")

        if config['test_neg_sampling_strategy'] == "random":
            start = time.time()
            print("Start: used_item copy and test collate_fn initialize...")
            cur_used_items = user_used_items['train'].copy()
            for user_id, u_items in user_used_items['validation'].items():
                cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
            for user_id, u_items in user_used_items['test'].items():
                cur_used_items[user_id] = cur_used_items[user_id].union(u_items)
            test_collate_fn = CollateNegSamplesRandomOpt(config['test_neg_samples'], cur_used_items,
                                                         padding_token=padding_token, joint=joint)
            print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")
        elif config['test_neg_sampling_strategy'].startswith("f:"):
            start = time.time()
            print("Start: used_item copy and test collate_fn initialize...")
            test_collate_fn = CollateOriginalDataPad(user_info, item_info, padding_token=padding_token, joint=joint)
            print(f"Finish: used_item copy and test collate_fn initialize {time.time() - start}")

        train_dataloader = DataLoader(datasets['train'],
                                      batch_size=config['train_batch_size'],
                                      shuffle=True,
                                      collate_fn=train_collate_fn,
                                      num_workers=config['dataloader_num_workers']
                                      )
        validation_dataloader = DataLoader(datasets['validation'],
                                           batch_size=config['eval_batch_size'],
                                           collate_fn=valid_collate_fn,
                                           num_workers=config['dataloader_num_workers'])
        test_dataloader = DataLoader(datasets['test'],
                                     batch_size=config['eval_batch_size'],
                                     collate_fn=test_collate_fn,
                                     num_workers=config['dataloader_num_workers'])
    return train_dataloader, validation_dataloader, test_dataloader, user_info, item_info, \
           config['relevance_level'] if 'relevance_level' in config else None, return_padding_token


class CollateRepresentationBuilder(object):
    def __init__(self, padding_token):
        self.padding_token = padding_token

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        ret = {}
        for col in ["chunks_input_ids", "chunks_attention_mask"]:
            if len(batch_df[col]) > 1:
                raise RuntimeError("precalc batch size must be 1")
            ret[col] = pad_sequence([torch.tensor(t) for t in batch_df[col][0]], batch_first=True,
                                    padding_value=self.padding_token).unsqueeze(1)
        for col in batch_df.columns:
            if col in ret:
                continue
            if col in ["user_id", "item_id"]:
                ret[col] = batch_df[col]
            else:
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


def jaccard_index(X, Y):
    d = len(X.intersection(Y))/len(X.union(Y))

    return d


class CollateNegSamplesRandomOpt(object):
    def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None, joint=False):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token
        self.joint = joint

    def sample(self, batch_df):
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in user_counter.keys():
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
                pass
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
        return samples

    def prepare_text_pad(self, batch_df):
        ret = {}
        if "chunks_input_ids" in self.user_info:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][
                ['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
                                                  "chunks_attention_mask": "user_chunks_attention_mask"})
            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][
                ['chunks_input_ids', 'chunks_attention_mask']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
                                                  "chunks_attention_mask": "item_chunks_attention_mask"})
            temp = pd.merge(temp_user[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label', 'user_chunks_input_ids', 'user_chunks_attention_mask']],
                            temp_item[[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD,  'label', 'item_chunks_input_ids', 'item_chunks_attention_mask']],
                            on=[INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD, 'label'])
            cols_to_pad = ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
                           "item_chunks_attention_mask"]
            if self.joint:
                temp["user_chunks_input_ids"] = temp["user_chunks_input_ids"].apply(lambda x: list(x[0]))
                temp["user_chunks_attention_mask"] = temp["user_chunks_attention_mask"].apply(lambda x: list(x[0]))
                temp["item_chunks_input_ids"] = temp["item_chunks_input_ids"].apply(lambda x: list(x[0])[1:])
                temp["item_chunks_attention_mask"] = temp["item_chunks_attention_mask"].apply(lambda x: list(x[0])[1:])
                temp["input_ids"] = temp["user_chunks_input_ids"] + temp["item_chunks_input_ids"]
                temp["attention_mask"] = temp["user_chunks_attention_mask"] + temp["item_chunks_attention_mask"]
                temp = temp.drop(columns=["user_chunks_input_ids", "user_chunks_attention_mask",
                                          "item_chunks_input_ids", "item_chunks_attention_mask"])
                cols_to_pad = ["attention_mask", "input_ids"]
                for col in cols_to_pad:
                    ret[col] = pad_sequence([torch.tensor(t) for t in temp[col]], batch_first=True, padding_value=self.padding_token)
            else:
                # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
                for col in cols_to_pad:
                    for i in range(len(temp[col])):
                        temp[col][i] = pad_sequence([torch.tensor(t) for t in temp[col][i]], batch_first=True, padding_value=self.padding_token)
                        # temp.loc[:, col].loc[i] = pad_sequence([torch.tensor(t) for t in temp.loc[:, col].loc[i]], batch_first=True, padding_value=self.padding_token)  #TODO make this better

                    max_len_0 = max([tensor.shape[0] for tensor in temp[col]])
                    max_len_1 = max([tensor.shape[1] for tensor in temp[col]])
                    padded = [torch.nn.functional.pad(tensor,
                                                      pad=(0, max_len_1 - tensor.shape[1], 0, max_len_0 - tensor.shape[0])) for tensor in temp[col]]
                    ret[col] = torch.stack(padded).transpose(0, 1)  # chunk * batch * dim
        elif "tokenized_text" in self.user_info:
            # user:
            temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['tokenized_text']] \
                .reset_index().drop(columns=['index'])
            temp_user = pd.concat([batch_df, temp_user], axis=1)
            temp_user = temp_user.rename(columns={"tokenized_text": "user_tokenized_text"})

            # item:
            temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['tokenized_text']] \
                .reset_index().drop(columns=['index'])
            temp_item = pd.concat([batch_df, temp_item], axis=1)
            temp_item = temp_item.rename(columns={"tokenized_text": "item_tokenized_text"})
            temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])
            cols_to_pad = ["user_tokenized_text", "item_tokenized_text"]
            # pad ,  the resulting tensor is batch * tokens -> bcs later we want to do batchwise
            for col in cols_to_pad:
                instances = [torch.tensor(instance) for instance in temp[col]]
                ret[col] = pad_sequence(instances, padding_value=self.padding_token, batch_first=True).type(
                    torch.int64)  # TODO token * bs
        for col in temp.columns:
            if col in ret:
                continue
            ret[col] = torch.tensor(temp[col]).unsqueeze(1)
        return ret

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        samples = self.sample(batch_df)
        batch_df = pd.concat([batch_df, pd.DataFrame(samples)]).reset_index().drop(columns=['index'])

        if self.padding_token is not None:
            ret = self.prepare_text_pad(batch_df)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateOriginalDataPad(CollateNegSamplesRandomOpt):
    def __init__(self, user_info, item_info, padding_token=None, joint=False):
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        self.padding_token = padding_token
        self.joint = joint

    def __call__(self, batch):
        batch_df = pd.DataFrame(batch)
        if self.padding_token is not None:
            ret = self.prepare_text_pad(batch_df)
        else:
            ret = {}
            for col in batch_df.columns:
                if col in ret:
                    continue
                ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
        return ret


class CollateNegSamplesRandomCFWeighted(CollateNegSamplesRandomOpt):
    def __init__(self, num_neg_samples, used_items, user_training_items,
                 cf_checkpoint_file, cf_item_id_file, oldmax, oldmin,
                 user_info=None, item_info=None, padding_token=None, joint=False):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        all_items = []
        for items in self.used_items.values():
            all_items.extend(items)
        self.all_items = list(set(all_items))
        self.padding_token = padding_token
        self.user_training_items = user_training_items
        temp = torch.load(cf_checkpoint_file, map_location=torch.device('cpu'))['model_state_dict']['item_embedding.weight']
        cf_item_internal_ids = json.load(open(cf_item_id_file, 'r'))
        item_info = item_info.to_pandas()
        self.cf_item_reps = {item_in: temp[cf_item_internal_ids[item_ex]] for item_ex, item_in in zip(item_info["item_id"], item_info[INTERNAL_ITEM_ID_FIELD])}
        self.oldmax = oldmax
        self.oldmin = oldmin
        if self.padding_token is not None:
            self.user_info = user_info.to_pandas()
            self.item_info = item_info
        self.joint = joint

    def sample(self, batch_df):
        user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
        samples = []
        for user_id in user_counter.keys():
            num_pos = user_counter[user_id]
            max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
            if max_num_user_neg_samples < num_pos * self.num_neg_samples:
                print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
                      f"but all_items are {len(self.all_items)}")
                pass
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
            # now we calculate the label weight and add the unlabeled samples to the samples list
            for sampled_item_id in user_samples:
                sims = []
                for pos_item in self.user_training_items[user_id]:
                    s = np.dot(self.cf_item_reps[sampled_item_id], self.cf_item_reps[pos_item])
                    s = (s - self.oldmin) / (self.oldmax - self.oldmin)  # s = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
                    # as oldmin and oldmax are estimates, we make sure that the s is between 0 and 1:
                    s = max(0, s)
                    s = min(1, s)
                    sims.append(s)
                avg_sim = sum(sims) / len(sims)
                samples.append({'label': avg_sim,
                                INTERNAL_USER_ID_FIELD: user_id,
                                INTERNAL_ITEM_ID_FIELD: sampled_item_id})
        return samples


# class CollateNegSamplesRandomOptJaccardWeightedLabels(CollateNegSamplesRandomOpt):
#     def __init__(self, num_neg_samples, used_items, user_training_items, item_user_set_file,
#                  user_info=None, item_info=None, padding_token=None):
#         self.num_neg_samples = num_neg_samples
#         self.used_items = used_items
#         # pool of all items is created from seen training items:
#         all_items = []
#         for items in self.used_items.values():
#             all_items.extend(items)
#         self.all_items = list(set(all_items))
#         self.user_info = user_info.to_pandas()
#         self.item_info = item_info.to_pandas()
#         self.item_info = self.item_info.set_index("item_id")
#         self.padding_token = padding_token
#         self.user_training_items = user_training_items
#         self.item_user_set = pickle.load(open(item_user_set_file, 'rb'))
#         self.item_user_set = {self.item_info.loc[k][INTERNAL_ITEM_ID_FIELD]: set(v) for k, v in self.item_user_set.items() if k in self.item_info.index}
#         self.item_info = self.item_info.reset_index()
#
#     def sample(self, batch_df):
#         user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
#         samples = []
#         for user_id in user_counter.keys():
#             num_pos = user_counter[user_id]
#             max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
#             if max_num_user_neg_samples < num_pos * self.num_neg_samples:
#                 print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
#                       f"but all_items are {len(self.all_items)}")
#                 pass
#             user_samples = set()
#             try_cnt = -1
#             num_user_neg_samples = max_num_user_neg_samples
#             while True:
#                 if try_cnt == 100:
#                     print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
#                           f"{user_id}. We instead have {len(user_samples)} samples.")
#                     break
#                 current_samples = set(random.sample(self.all_items, num_user_neg_samples))
#                 current_samples -= user_samples
#                 cur_used_samples = self.used_items[user_id].intersection(current_samples)
#                 current_samples = current_samples - cur_used_samples
#                 user_samples = user_samples.union(current_samples)
#                 num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
#                 if len(user_samples) < max_num_user_neg_samples:
#                     # to make the process faster
#                     if num_user_neg_samples < len(user_samples):
#                         num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
#                     try_cnt += 1
#                 else:
#                     if len(user_samples) > max_num_user_neg_samples:
#                         user_samples = set(list(user_samples)[:max_num_user_neg_samples])
#                     break
#             # now we calculate the label weight and add the unlabeled samples to the samples list
#             for sampled_item_id in user_samples:
#                 relatedness = [jaccard_index(self.item_user_set[sampled_item_id], self.item_user_set[pos_item])
#                                for pos_item in self.user_training_items[user_id]]  # todo entire user_training_items? or user items in this batch?
#                 avg_relatedness = sum(relatedness) / len(relatedness)
#                 samples.append({'label': avg_relatedness,
#                                 INTERNAL_USER_ID_FIELD: user_id,
#                                 INTERNAL_ITEM_ID_FIELD: sampled_item_id})
#         return samples


# class CollateNegSamplesRandomOptClassPriorWeightedLabels(CollateNegSamplesRandomOpt):
#     def __init__(self, num_neg_samples, used_items, pos_class_prior, user_info=None, item_info=None, padding_token=None):
#         self.num_neg_samples = num_neg_samples
#         self.used_items = used_items
#         # pool of all items is created from seen training items:
#         all_items = []
#         for items in self.used_items.values():
#             all_items.extend(items)
#         self.all_items = list(set(all_items))
#         self.user_info = user_info.to_pandas()
#         self.item_info = item_info.to_pandas()
#         self.padding_token = padding_token
#         self.pos_class_prior = pos_class_prior
#
#     def sample(self, batch_df):
#         user_counter = Counter(batch_df[INTERNAL_USER_ID_FIELD])
#         samples = []
#         for user_id in user_counter.keys():
#             num_pos = user_counter[user_id]
#             max_num_user_neg_samples = min(len(self.all_items), num_pos * self.num_neg_samples)
#             if max_num_user_neg_samples < num_pos * self.num_neg_samples:
#                 print(f"WARN: user {user_id} needed {num_pos * self.num_neg_samples} samples,"
#                       f"but all_items are {len(self.all_items)}")
#                 pass
#             user_samples = set()
#             try_cnt = -1
#             num_user_neg_samples = max_num_user_neg_samples
#             while True:
#                 if try_cnt == 100:
#                     print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
#                           f"{user_id}. We instead have {len(user_samples)} samples.")
#                     break
#                 current_samples = set(random.sample(self.all_items, num_user_neg_samples))
#                 current_samples -= user_samples
#                 cur_used_samples = self.used_items[user_id].intersection(current_samples)
#                 current_samples = current_samples - cur_used_samples
#                 user_samples = user_samples.union(current_samples)
#                 num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
#                 if len(user_samples) < max_num_user_neg_samples:
#                     # to make the process faster
#                     if num_user_neg_samples < len(user_samples):
#                         num_user_neg_samples = min(max_num_user_neg_samples, num_user_neg_samples * 2)
#                     try_cnt += 1
#                 else:
#                     if len(user_samples) > max_num_user_neg_samples:
#                         user_samples = set(list(user_samples)[:max_num_user_neg_samples])
#                     break
#             samples.extend([{'label': self.pos_class_prior,
#                              INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
#                             for sampled_item_id in user_samples])
#         return samples


# class CollateNegSamplesRandomOptOrdered(CollateNegSamplesRandomOpt):
#     def __init__(self, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None):
#         self.num_neg_samples = num_neg_samples
#         self.used_items = used_items
#         # pool of all items is created from seen training items:
#         all_items = []
#         for items in self.used_items.values():
#             all_items.extend(items)
#         self.all_items = list(set(all_items))
#         self.user_info = user_info.to_pandas()
#         self.item_info = item_info.to_pandas()
#         self.padding_token = padding_token
#
#     def __call__(self, batch):
#         samples = []
#         total_user_samples = defaultdict(set)
#         for b in batch:
#             user_id = b[INTERNAL_USER_ID_FIELD]
#             max_num_user_neg_samples = min(len(self.all_items), self.num_neg_samples)
#             try_cnt = -1
#             num_user_neg_samples = max_num_user_neg_samples
#             user_samples = set()
#             while True:
#                 if try_cnt == 100:
#                     # print(f"WARN: After {try_cnt} tries, could not find {max_num_user_neg_samples} samples for"
#                     #       f"{user_id}. We instead have {len(user_samples)} samples.")
#                     break
#                 current_samples = set(random.sample(self.all_items, num_user_neg_samples))
#                 current_samples -= user_samples
#                 current_samples -= total_user_samples[user_id]
#                 current_samples -= self.used_items[user_id]
#                 user_samples.update(current_samples)
#                 num_user_neg_samples = max(max_num_user_neg_samples - len(user_samples), 0)
#                 if len(user_samples) < max_num_user_neg_samples:
#                     try_cnt += 1
#                 else:
#                     if len(user_samples) > max_num_user_neg_samples:
#                         user_samples = set(list(user_samples)[:max_num_user_neg_samples])
#                     break
#             total_user_samples[user_id].update(user_samples)
#             samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
#                             for sampled_item_id in user_samples])
#         batch_df = pd.concat([pd.DataFrame(batch), pd.DataFrame(samples)]).reset_index().drop(columns=['index'])
#
#         # todo make this somehow that each of them could have text and better code
#         if self.padding_token is not None:
#             # user:
#             temp_user = self.user_info.loc[batch_df[INTERNAL_USER_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
#                 .reset_index().drop(columns=['index'])
#             temp_user = pd.concat([batch_df, temp_user], axis=1)
#             temp_user = temp_user.rename(columns={"chunks_input_ids": "user_chunks_input_ids",
#                                                   "chunks_attention_mask": "user_chunks_attention_mask"})
#             # item:
#             temp_item = self.item_info.loc[batch_df[INTERNAL_ITEM_ID_FIELD]][['chunks_input_ids', 'chunks_attention_mask']] \
#                 .reset_index().drop(columns=['index'])
#             temp_item = pd.concat([batch_df, temp_item], axis=1)
#             temp_item = temp_item.rename(columns={"chunks_input_ids": "item_chunks_input_ids",
#                                                   "chunks_attention_mask": "item_chunks_attention_mask"})
#             temp = pd.merge(temp_user, temp_item, on=['label', 'internal_user_id', 'internal_item_id'])
#
#             # pad ,  the resulting tensor is num-chunks * batch * tokens -> bcs later we want to do batchwise
#             ret = {}
#             for col in ["user_chunks_input_ids", "user_chunks_attention_mask", "item_chunks_input_ids",
#                         "item_chunks_attention_mask"]:
#                 # instances = [pad_sequence([torch.tensor(t) for t in instance], padding_value=self.padding_token) for
#                 #              instance in temp[col]]
#                 instances = [torch.tensor([list(t) for t in instance]) for instance in temp[col]]
#                 ret[col] = pad_sequence(instances, padding_value=self.padding_token).type(torch.int64)
#             for col in temp.columns:
#                 if col in ret:
#                     continue
#                 ret[col] = torch.tensor(temp[col]).unsqueeze(1)
#         else:
#             ret = {}
#             for col in batch_df.columns:
#                 if col in ret:
#                     continue
#                 ret[col] = torch.tensor(batch_df[col]).unsqueeze(1)
#         return ret


class CollateNegSamplesGenresOpt(CollateNegSamplesRandomOpt):
    def __init__(self, strategy, num_neg_samples, used_items, user_info=None, item_info=None, padding_token=None, joint=False):
        self.num_neg_samples = num_neg_samples
        self.used_items = used_items
        # pool of all items is created from seen training items:
        self.user_info = user_info.to_pandas()
        self.item_info = item_info.to_pandas()
        genres_field = 'category' if 'category' in self.item_info.columns else 'genres'
        print("start parsing item genres")
        self.genre_items = defaultdict(set)
        self.item_genres = defaultdict(list)
        for item, genres in zip(self.item_info[INTERNAL_ITEM_ID_FIELD], self.item_info[genres_field]):
            for g in [g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").strip() for g in genres.split(",")]:
                self.genre_items[g].add(item)
                self.item_genres[item].append(g)
        for g in self.genre_items:
            self.genre_items[g] = list(self.genre_items[g])
        print("finish parsing item genres")
        self.joint = joint

        # if strategy == "genres":
        #     print("start creating item candidates")
        #     self.item_candidates = defaultdict(list)
        #     for item, genres in item_genres.items():
        #         for g in genres:
        #             self.item_candidates[item].extend(list(genres_item[g] - set([item])))
        #         self.item_candidates[item] = Counter(self.item_candidates[item])
        #     print("finish creating item candidates")

        self.strategy = strategy
        self.padding_token = padding_token

    def sample(self, batch_df):
        samples = []
        for user_id, item_id in zip(batch_df[INTERNAL_USER_ID_FIELD], batch_df[INTERNAL_ITEM_ID_FIELD]):
            sampled_genres = []
            while len(sampled_genres) < self.num_neg_samples:
                sampled_genres.extend(np.random.choice(self.item_genres[item_id],
                                                       min(self.num_neg_samples - len(sampled_genres),
                                                           len(self.item_genres[item_id])),
                                                       replace=False).tolist())
            neg_samples = set()
            for g in sampled_genres:
                try_cnt = -1
                while True:
                    if try_cnt == 20:
                        break
                    s = random.choice(self.genre_items[g])
                    if s not in neg_samples and s not in self.used_items[user_id]:
                        neg_samples.add(s)
                        break
                    try_cnt += 1
            samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
                            for sampled_item_id in neg_samples])
        return samples

    # def sample(self, batch_df):
    #     samples = []
    #     if self.strategy == "genres":
    #         user_samples = defaultdict(set)
    #         for user_id, item_id in zip(batch_df[INTERNAL_USER_ID_FIELD], batch_df[INTERNAL_ITEM_ID_FIELD]):
    #             # create item candidate on the fly, as it doesn't fit memory
    #             item_candidates = []
    #             for g in self.item_genres[item_id]:
    #                 item_candidates.extend(self.genres_items[g])
    #             item_candidates = Counter(item_candidates)
    #             item_candidates.pop(item_id)
    #             candids = {k: v for k, v in item_candidates.items()
    #                        if (k not in self.used_items[user_id] and k not in user_samples[user_id])}
    #
    #             candids = {k: v for k, v in self.item_candidates[item_id].items()
    #                        if (k not in self.used_items[user_id] and k not in user_samples[user_id])}
    #             sum_w = sum(candids.values())
    #             if sum_w > 0:
    #                 sampled_item_ids = np.random.choice(list(candids.keys()), min(len(candids), self.num_neg_samples),
    #                                                     p=[c / sum_w for c in candids.values()], replace=False)
    #                 samples.extend([{'label': 0, INTERNAL_USER_ID_FIELD: user_id, INTERNAL_ITEM_ID_FIELD: sampled_item_id}
    #                                 for sampled_item_id in sampled_item_ids])
    #                 user_samples[user_id].update(set(sampled_item_ids))
    #     return samples


def get_user_used_items(datasets, filtered_out_user_item_pairs_by_limit):
    used_items = {}
    for split in datasets.keys():
        used_items[split] = defaultdict(set)
        for user_iid, item_iid in zip(datasets[split][INTERNAL_USER_ID_FIELD], datasets[split][INTERNAL_ITEM_ID_FIELD]):
            used_items[split][user_iid].add(item_iid)

    for user, books in filtered_out_user_item_pairs_by_limit.items():
        used_items['train'][user] = used_items['train'][user].union(books)

    return used_items


def load_split_dataset(config, for_precalc=False):
    user_text_file_name = config['user_text_file_name']
    item_text_file_name = config['item_text_file_name']
    if config['load_user_item_text'] is False:
        user_text_file_name = None
        item_text_file_name = None

    keep_fields = ["user_id"]
    user_info = pd.read_csv(join(config['dataset_path'], "users.csv"), usecols=keep_fields, dtype=str)
    user_info = user_info.sort_values("user_id").reset_index(drop=True) # this is crucial, as the precomputing is done with internal ids
    user_info[INTERNAL_USER_ID_FIELD] = np.arange(0, user_info.shape[0])
    if len(user_info["user_id"]) != len(set(user_info["user_id"])):
        raise ValueError("problem in users.csv file")
    print(f"num users = {len(user_info[INTERNAL_USER_ID_FIELD])}")
    up = pd.read_csv(join(config['dataset_path'], f"users_profile_{user_text_file_name}.csv"), dtype=str)
    up = up.fillna('')
    if len(up["user_id"]) != len(set(user_info["user_id"])):
        raise ValueError(f"problem in users_profile_{user_text_file_name}.csv file")
    user_info = pd.merge(user_info, up, on="user_id")
    user_info['text'] = user_info['text'].apply(lambda x: x.replace("<end of review>", ""))
    if not config['case_sensitive']:
        user_info['text'] = user_info['text'].apply(str.lower)
    if config['normalize_negation']:
        user_info['text'] = user_info['text'].replace("n\'t", " not", regex=True)

    keep_fields = ["item_id"]
    if not for_precalc and config["training_neg_sampling_strategy"] == "genres":  # TODO if we added genres_weighted...
        if config["name"] == "Amazon":
            keep_fields.append("category")
        elif config["name"] in ["CGR", "GR_UCSD"]:
            keep_fields.append("genres")
        else:
            raise NotImplementedError()
    item_info = pd.read_csv(join(config['dataset_path'], "items.csv"), usecols=keep_fields, low_memory=False, dtype=str)
    item_info = item_info.sort_values("item_id").reset_index(drop=True)  # this is crucial, as the precomputing is done with internal ids
    item_info[INTERNAL_ITEM_ID_FIELD] = np.arange(0, item_info.shape[0])
    if len(item_info["item_id"]) != len(set(item_info["item_id"])):
        raise ValueError("problem in items.csv file")
    print(f"num items = {len(item_info[INTERNAL_ITEM_ID_FIELD])}")
    item_info = item_info.fillna('')
    ip = pd.read_csv(join(config['dataset_path'], f"item_profile_{item_text_file_name}.csv"), dtype=str)
    ip = ip.fillna('')
    if len(ip["item_id"]) != len(set(item_info["item_id"])):
        raise ValueError(f"problem in item_profile_{item_text_file_name}.csv file")
    item_info = pd.merge(item_info, ip, on="item_id")
    item_info['text'] = item_info['text'].apply(lambda x: x.replace("<end of review>", ""))
    if not config['case_sensitive']:
        item_info['text'] = item_info['text'].apply(str.lower)
    if config['normalize_negation']:
        item_info['text'] = item_info['text'].replace("n\'t", " not", regex=True)


    if 'item.genres' in item_info.columns:
        item_info['item.genres'] = item_info['item.genres'].apply(
            lambda x: ", ".join([g.replace("'", "").replace('"', "").replace("[", "").replace("]", "").replace("  ", " ").strip() for
                                 g in x.split(",")]))
    if config["name"] == "Amazon":
        if 'item.category' in item_info.columns:
            item_info['item.category'] = item_info['item.category'].apply(
                lambda x: ", ".join(x[1:-1].split(",")).replace("'", "").replace('"', "").replace("  ", " ")
                .replace("[", "").replace("]", "").strip())

    # read user-item interactions, map the user and item ids to the internal ones
    sp_files = {"train": join(config['dataset_path'], "train.csv"),
                "validation": join(config['dataset_path'], f'{config["alternative_pos_validation_file"]}.csv' if ("alternative_pos_validation_file" in config and config["alternative_pos_validation_file"] != "") else "validation.csv"),
                "test": join(config['dataset_path'], "test.csv")}
    split_datasets = {}
    filtered_out_user_item_pairs_by_limit = defaultdict(set)
    for sp, file in sp_files.items():
        df = pd.read_csv(file, usecols=["user_id", "item_id", "rating"], dtype=str)  # rating:float64   TODO label?
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

        # TODO if dataset is cleaned beforehand this could change slightly
        df['rating'] = df['rating'].fillna(-1)
        if config["name"] == "CGR":
            for k, v in goodreads_rating_mapping.items():
                df['rating'] = df['rating'].replace(k, v)
        elif config["name"] == "GR_UCSD":
            df['rating'] = df['rating'].astype(int)
        elif config["name"] == "Amazon":
            df['rating'] = df['rating'].astype(float).astype(int)
        else:
            raise NotImplementedError(f"dataset {config['name']} not implemented!")
        if not for_precalc:
            if config['binary_interactions']:
                # if binary prediction (interaction): set label for all interactions to 1.
                df['label'] = np.ones(df.shape[0])
            else:
                # if predicting rating: remove the not-rated entries and map rating text to int
                df = df[df['rating'] != -1].reset_index()
                df['label'] = df['rating']

        # replace user_id with internal_user_id (same for item_id)
        df = df.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        df = df.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        df = df.drop(columns=["user_id", "item_id"])
        split_datasets[sp] = df

    if not for_precalc and config["training_neg_sampling_strategy"] == "genres":  # TODO if we added genres_weighted...
        if config["name"] == "Amazon":
            if 'category' not in item_info.columns:
                item_info['category'] = item_info['item.category']
        elif config["name"] in ["CGR", "GR_UCSD"]:
            if 'genres' not in item_info.columns:
                item_info['genres'] = item_info['item.genres']
        else:
            raise NotImplementedError()

    # loading negative samples for eval sets: I used to load them in a collatefn, but, because batch=101 does not work for evaluation for BERT-based models
    if not for_precalc and config['validation_neg_sampling_strategy'].startswith("f:"):
        fname = config['validation_neg_sampling_strategy'][2:]
        negs = pd.read_csv(join(config['dataset_path'], fname+".csv"), dtype=str)
        negs['label'] = 0  # used to have weightes for weighted evaluation, but then removed that code
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        if "ref_item" in negs.columns:
            negs = negs.drop(columns=["ref_item"])
        split_datasets['validation'] = pd.concat([split_datasets['validation'], negs])
        split_datasets['validation'] = split_datasets['validation'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    if not for_precalc and config['test_neg_sampling_strategy'].startswith("f:"):
        fname = config['test_neg_sampling_strategy'][2:]
        negs = pd.read_csv(join(config['dataset_path'], fname+".csv"), dtype=str)
        negs['label'] = 0  # used to have weightes for weighted evaluation, but then removed that code
        negs = negs.merge(user_info[["user_id", INTERNAL_USER_ID_FIELD]], "left", on="user_id")
        negs = negs.merge(item_info[["item_id", INTERNAL_ITEM_ID_FIELD]], "left", on="item_id")
        negs = negs.drop(columns=["user_id", "item_id"])
        if "ref_item" in negs.columns:
            negs = negs.drop(columns=["ref_item"])        
        split_datasets['test'] = pd.concat([split_datasets['test'], negs])
        split_datasets['test'] = split_datasets['test'].sort_values(INTERNAL_USER_ID_FIELD).reset_index().drop(columns=['index'])

    for split in split_datasets.keys():
        split_datasets[split] = Dataset.from_pandas(split_datasets[split], preserve_index=False)

    return DatasetDict(split_datasets), Dataset.from_pandas(user_info, preserve_index=False), \
           Dataset.from_pandas(item_info, preserve_index=False), filtered_out_user_item_pairs_by_limit

