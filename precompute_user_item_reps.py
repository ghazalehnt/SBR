import argparse
import json
import os
import time

import torch
import numpy as np
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.data_loading import load_data, CollateRepresentationBuilder
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


def main(config_file, given_user_text_filter=None):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = json.load(open(config_file, 'r'))
    if given_user_text_filter is not None:
        config['dataset']['user_text_filter'] = given_user_text_filter
    if config['model']['precalc_batch_size'] > 1:
        raise ValueError("There is a bug when the batch size is bigger than one. Users/items with only one chunk"
                         "are producing wrong reps. Please set the batch size to 1.")

    if "<DATA_ROOT_PATH>" in config["dataset"]["dataset_path"]:
        config["dataset"]["dataset_path"] = config["dataset"]["dataset_path"] \
            .replace("<DATA_ROOT_PATH>", open("data/paths_vars/DATA_ROOT_PATH").read().strip())

    train_dataloader, valid_dataloader, test_dataloader, users, items, relevance_level, padding_token = \
        load_data(config['dataset'],
                  config['model']['pretrained_model'] if 'pretrained_model' in config['model'] else None)

    prec_path = os.path.join(config['dataset']['dataset_path'], 'precomputed_reps',
                             f"size{config['dataset']['chunk_size']}_"
                             f"cs-{config['dataset']['case_sensitive']}_"
                             f"nn-{config['dataset']['normalize_negation']}_"
                             f"u{config['dataset']['max_num_chunks_user']}-"
                             f"{'-'.join(config['dataset']['user_text'])}_{config['dataset']['user_review_choice']}_"
                             f"{config['dataset']['review_tie_breaker'] if len(config['dataset']['user_text_filter']) == 0 else ''}_"
                             f"{config['dataset']['user_text_filter'] if len(config['dataset']['user_text_filter']) > 0 else 'no-filter'}_"
                             f"i{config['dataset']['max_num_chunks_item']}-{'-'.join(config['dataset']['item_text'])}")
    print(prec_path)
    os.makedirs(prec_path, exist_ok=True)

    agg_strategy = config['model']['agg_strategy']
    chunk_agg_strategy = config['model']['chunk_agg_strategy']
    batch_size = config['model']['precalc_batch_size']

    bert = transformers.AutoModel.from_pretrained(config['model']['pretrained_model'])
    bert.to(device)  # need to move to device earlier as we are precalculating.
    if config['model']['tune_BERT'] is False:
        for param in bert.parameters():
            param.requires_grad = False
    else:
        raise ValueError("You cannot precompute if you want to tune BERT!")
    bert_embedding_dim = bert.embeddings.word_embeddings.weight.shape[1]
    bert_embeddings = bert.get_input_embeddings()

    user_id_embedding, item_id_embedding = None, None
    if config['model']["append_id"]:
        user_id_embedding = torch.nn.Embedding(users.shape[0], bert_embedding_dim, device=device)
        item_id_embedding = torch.nn.Embedding(items.shape[0], bert_embedding_dim, device=device)

    user_embedding_CF, item_embedding_CF = None, None
    if config['model']['use_CF']:
        # loading the pretrained CF embeddings for users and items
        CF_model_weights = torch.load(config['model']['CF_model_path'], map_location=device)['model_state_dict']
        user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'])
        item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'])

    start = time.time()
    user_rep_file = f"{agg_strategy}_{chunk_agg_strategy}_" \
                    f"id{config['model']['append_id']}_tb{config['model']['tune_BERT']}_" \
                    f"cf{config['model']['use_CF']}_user_representation.pkl"
    if os.path.exists(os.path.join(prec_path, user_rep_file)):
        print(f"EXISTED ALREADY, NOT CREATED: \n{os.path.join(prec_path, user_rep_file)}")
    else:
        weights = create_representations(bert, bert_embeddings, users, padding_token, device, batch_size, agg_strategy,
                                         chunk_agg_strategy, config['dataset']['dataloader_num_workers'], INTERNAL_USER_ID_FIELD,
                                         user_id_embedding if config['model']['append_id'] else None,
                                         user_embedding_CF if config['model']['use_CF'] else None)
        torch.save(weights, os.path.join(prec_path, user_rep_file))
    print(f"user rep created  {time.time() - start}")

    start = time.time()
    item_rep_file = f"{agg_strategy}_{chunk_agg_strategy}_" \
                    f"id{config['model']['append_id']}_tb{config['model']['tune_BERT']}_" \
                    f"cf{config['model']['use_CF']}_item_representation.pkl"
    if os.path.exists(os.path.join(prec_path, item_rep_file)):
        print(f"EXISTED ALREADY, NOT CREATED: \n{os.path.join(prec_path, item_rep_file)}")
    else:
        weights = create_representations(bert, bert_embeddings, items, padding_token, device, batch_size, agg_strategy,
                                         chunk_agg_strategy, config['dataset']['dataloader_num_workers'], INTERNAL_ITEM_ID_FIELD,
                                         item_id_embedding if config['model']['append_id'] else None,
                                         item_embedding_CF if config['model']['use_CF'] else None)
        torch.save(weights, os.path.join(prec_path, item_rep_file))
        print(f"item rep created in {time.time() - start}")


def create_representations(bert, bert_embeddings, info, padding_token, device, batch_size, agg_strategy,
                           chunk_agg_strategy, num_workers, id_field, id_embedding=None, embedding_CF=None):
    collate_fn = CollateRepresentationBuilder(padding_token=padding_token)
    dataloader = DataLoader(info, batch_size=batch_size, collate_fn=collate_fn, num_workers=num_workers)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    reps = []
    for batch_idx, batch in pbar:
        # go over chunks:
        outputs = []
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
                token_embeddings = bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask],
                                            dim=1)
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
                # summed_mask = torch.clamp(mask.sum(1), min=1e-9)  -> I see the point, but it's better to leave it as is to find the errors as in our case there should be something
                temp = sum_tokons / mask.sum(1)
            else:
                raise ValueError(f"agg_strategy not implemented {agg_strategy}")
            outputs.append(temp)
        if chunk_agg_strategy == "max_pool":
            rep = torch.stack(outputs).max(dim=0).values
        else:
            raise ValueError("not implemented")
        reps.append(rep)
    return torch.concat(reps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default=None, help='config file, to train')
    parser.add_argument('--user_text_filter', type=str, default=None, help='user_text_filter used only if given, otherwise read from the config')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.config_file):
        raise ValueError(f"Config file does not exist: {args.config_file}")
    main(config_file=args.config_file, given_user_text_filter=args.user_text_filter)
    

