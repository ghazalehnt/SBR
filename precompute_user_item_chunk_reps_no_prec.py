import argparse
import json
import os
import time
from collections import defaultdict

import torch
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.data_loading import load_data, CollateRepresentationBuilder
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

BERT_DIM = 768


def main(config_file, given_limit_training_data=None,
         given_user_text_file_name=None, given_item_text_file_name=None, calc_which=None):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open(config_file, 'r'))
    if given_limit_training_data is not None:
        config['dataset']['limit_training_data'] = given_limit_training_data
    if given_user_text_file_name is not None:
        config['dataset']['user_text_file_name'] = given_user_text_file_name
    if given_item_text_file_name is not None:
        config['dataset']['item_text_file_name'] = given_item_text_file_name
    if config['model']['precalc_batch_size'] > 1:
        raise ValueError("There is a bug when the batch size is bigger than one. Users/items with only one chunk"
                         "are producing wrong reps. Please set the batch size to 1.")

    if "<DATA_ROOT_PATH" in config["dataset"]["dataset_path"]:
        DATA_ROOT_PATH = config["dataset"]["dataset_path"][config["dataset"]["dataset_path"].index("<"):
                                                           config["dataset"]["dataset_path"].index(">") + 1]
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"] \
            .replace(DATA_ROOT_PATH, open(f"data/paths_vars/{DATA_ROOT_PATH[1:-1]}").read().strip())

    if 'max_num_chunks_user' in config['dataset'] or 'max_num_chunks_item' in config['dataset']:
        raise ValueError("max num chunks should not be set")
    if 'chunk_agg_strategy' in config['model']:
        raise ValueError('chunk_agg_strategy should not ne set')

    _, _, _, users, items, _, padding_token = \
        load_data(config['dataset'],
                  pretrained_model=config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None,
                  for_precalc=True)
    print("Data load done!")
    agg_strategy = config['model']['agg_strategy']
    batch_size = config['model']['precalc_batch_size']
    bert = transformers.AutoModel.from_pretrained(config['model']['pretrained_model'])
    bert.requires_grad_(False)
    bert.to(device)  # need to move to device earlier as we are precalculating.
    bert_embedding_dim = bert.embeddings.word_embeddings.weight.shape[1]
    bert_embeddings = bert.get_input_embeddings()

    CF_model_weights = None
    if config['model']['use_CF']:
        CF_model_weights = torch.load(config['model']['CF_model_path'], map_location=device)['model_state_dict']
        embedding_dim = CF_model_weights['user_embedding.weight'].shape[-1]
        if embedding_dim > BERT_DIM:
            raise ValueError("The CF embedding cannot be bigger than BERT dim")
        elif embedding_dim < BERT_DIM:
            print("CF embedding dim was smaller than BERT dim, therefore will be filled with  0's.")

    if calc_which is None or calc_which == "user":
        user_prec_path = os.path.join(config['dataset']['dataset_path'], f'precomputed_reps{f"_MF-{embedding_dim}" if CF_model_weights is not None else ""}',
                                      f"size{config['dataset']['user_chunk_size']}_"
                                      f"cs-{config['dataset']['case_sensitive']}_"
                                      f"nn-{config['dataset']['normalize_negation']}_"
                                      f"{config['dataset']['limit_training_data'] if len(config['dataset']['limit_training_data']) > 0 else 'no-limit'}")
        print(user_prec_path)
        os.makedirs(user_prec_path, exist_ok=True)
        user_id_embedding = None
        if config['model']["append_id"]:
            user_id_embedding = torch.nn.Embedding(users.shape[0], bert_embedding_dim, device=device)
        user_embedding_CF = None
        if config['model']['use_CF']:
            user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'])

        start = time.time()
        user_rep_file = f"user_representation_" \
                        f"{agg_strategy}_" \
                        f"id{config['model']['append_id']}_" \
                        f"cf{config['model']['use_CF']}_" \
                        f"{config['dataset']['user_text_file_name']}" \
                        f".pkl"

        if os.path.exists(os.path.join(user_prec_path, user_rep_file)):
            print(f"EXISTED ALREADY, NOT CREATED: \n{os.path.join(user_prec_path, user_rep_file)}")
        else:
            weights = create_representations(bert, bert_embeddings, users, padding_token, device, batch_size,
                                             agg_strategy,
                                             config['dataset']['dataloader_num_workers'],
                                             INTERNAL_USER_ID_FIELD, "user_id",
                                             user_id_embedding if config['model']['append_id'] else None,
                                             user_embedding_CF if config['model']['use_CF'] else None)
            torch.save(weights, os.path.join(user_prec_path, user_rep_file))
        print(f"user rep created  {time.time() - start}")

    if calc_which is None or calc_which == "item":
        item_prec_path = os.path.join(config['dataset']['dataset_path'], f'precomputed_reps{f"_MF-{embedding_dim}" if CF_model_weights is not None else ""}',
                                      f"size{config['dataset']['item_chunk_size']}_"
                                      f"cs-{config['dataset']['case_sensitive']}_"
                                      f"nn-{config['dataset']['normalize_negation']}_"
                                      f"{config['dataset']['limit_training_data'] if len(config['dataset']['limit_training_data']) > 0 else 'no-limit'}")
        print(item_prec_path)
        os.makedirs(item_prec_path, exist_ok=True)
        item_id_embedding = None
        if config['model']["append_id"]:
            item_id_embedding = torch.nn.Embedding(items.shape[0], bert_embedding_dim, device=device)
        item_embedding_CF = None
        if config['model']['use_CF']:
            item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'])

        start = time.time()
        item_rep_file = f"item_representation_" \
                        f"{agg_strategy}_" \
                        f"id{config['model']['append_id']}_" \
                        f"cf{config['model']['use_CF']}_" \
                        f"{'-'.join(config['dataset']['item_text_file_name'])}" \
                        f".pkl"
        if os.path.exists(os.path.join(item_prec_path, item_rep_file)):
            print(f"EXISTED ALREADY, NOT CREATED: \n{os.path.join(item_prec_path, item_rep_file)}")
        else:
            weights = create_representations(bert, bert_embeddings, items, padding_token, device, batch_size,
                                             agg_strategy,
                                             config['dataset']['dataloader_num_workers'],
                                             INTERNAL_ITEM_ID_FIELD, "item_id",
                                             item_id_embedding if config['model']['append_id'] else None,
                                             item_embedding_CF if config['model']['use_CF'] else None)
            torch.save(weights, os.path.join(item_prec_path, item_rep_file))
        print(f"item rep created in {time.time() - start}")


def create_representations(bert, bert_embeddings, info, padding_token, device, batch_size, agg_strategy,
                           num_workers, id_field, external_id_field, id_embedding=None, embedding_CF=None):
    collate_fn = CollateRepresentationBuilder(padding_token=padding_token)
    dataloader = DataLoader(info, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    reps = defaultdict()
    # reps = []
    for batch_idx, batch in pbar:
        # go over chunks:
        outputs = []
        ex_id = batch[external_id_field].values[0]
        ids = batch[id_field].to(device)
        for input_ids, att_mask in zip(batch['chunks_input_ids'], batch['chunks_attention_mask']):
            input_ids = input_ids.to(device)
            att_mask = att_mask.to(device)
            if id_embedding is not None and embedding_CF is not None:
                id_embeds = id_embedding(ids)
                cf_embeds = embedding_CF(ids)
                token_embeddings = bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, id_embeds, cf_embeds], dim=1), other_tokens],
                                          dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 2), device=att_mask.device), att_mask],
                                            dim=1)
                output = bert.forward(inputs_embeds=concat_ids,
                                      attention_mask=att_mask)
            elif id_embedding is not None:
                id_embeds = id_embedding(ids)
                token_embeddings = bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, id_embeds], dim=1), other_tokens], dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask],
                                            dim=1)
                output = bert.forward(inputs_embeds=concat_ids,
                                      attention_mask=att_mask)
            elif embedding_CF is not None:
                cf_embeds = embedding_CF(ids)
                if embedding_CF.embedding_dim < BERT_DIM:
                    cf_embeds = torch.concat([cf_embeds, torch.zeros((cf_embeds.shape[0], cf_embeds.shape[1], BERT_DIM-cf_embeds.shape[2]), device=cf_embeds.device)], dim=2)
                token_embeddings = bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]

                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask], dim=1)
                output = bert.forward(inputs_embeds=concat_ids,
                                      attention_mask=att_mask)
            else:
                output = bert.forward(input_ids=input_ids,
                                      attention_mask=att_mask)
            if agg_strategy == "CLS":
                temp = output.pooler_output
            elif agg_strategy == "mean_last":
                tokens_embeddings = output.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
                temp = (sum_tokons.T / summed_mask).T  # divide by how many tokens (1s) are in the att_mask
            else:
                raise ValueError(f"agg_strategy not implemented {agg_strategy}")
            outputs.append(temp.to('cpu'))
        reps[ex_id] = outputs
        # reps.append(outputs)
    return reps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--limit_training_data', '-l', type=str, default=None,
                        help='the file name containing the limited training data')
    parser.add_argument('--user_text_file_name', default=None, help='user_text_file_name')
    parser.add_argument('--item_text_file_name', default=None, help='item_text_file_name')
    parser.add_argument('--which', default=None, help='if specified, only calculate user/item reps otherwhise both.')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.config_file):
        raise ValueError(f"Config file does not exist: {args.config_file}")
    main(config_file=args.config_file,
         given_limit_training_data=args.limit_training_data,
         given_user_text_file_name=args.user_text_file_name, given_item_text_file_name=args.item_text_file_name,
         calc_which=args.which)
