import os.path
import pickle
import time

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.data_loading import CollateRepresentationBuilder


class VanillaClassifierUserTextProfileItemTextProfilePrecalculated(torch.nn.Module):
    def __init__(self, config, n_users, n_items, num_classes, user_info, item_info, padding_token, device, prec_dir):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculated, self).__init__()
        bert = transformers.AutoModel.from_pretrained(config['pretrained_model'])
        bert.to(device)  # need to move to device earlier as we are precalculating.

        if config['tune_BERT'] is False:
            for param in bert.parameters():
                param.requires_grad = False
        bert_embedding_dim = bert.embeddings.word_embeddings.weight.shape[1]

        # it is better to always freeze the reps, and not append a random ID... so freeze_prec_reps:True, append_id:False
        self.freeze_prec_rep = config['freeze_prec_reps']
        bert_embeddings = None
        if config["append_id"]:
            self.user_id_embedding = torch.nn.Embedding(n_users, bert_embedding_dim, device=device)
            if self.freeze_prec_rep:
                self.user_id_embedding.requires_grad_(False)
            self.item_id_embedding = torch.nn.Embedding(n_items, bert_embedding_dim, device=device)
            if self.freeze_prec_rep:
                self.item_id_embedding.requires_grad_(False)
            bert_embeddings = bert.get_input_embeddings()

        if "k" in config and config["k"] not in ['', 0]:
            self.transform_u = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.transform_i = torch.nn.Linear(bert_embedding_dim, config['k'])

        if config['similarity'] == "dot_product":
            self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
            self.item_bias = torch.nn.Parameter(torch.zeros(n_items))
            self.bias = torch.nn.Parameter(torch.zeros(1))
        elif config['similarity'] == "classifier":
            if "k" in config and config["k"] not in ['', 0]:
                self.classifier = torch.nn.Linear(2 * config['k'], num_classes)
            else:
                self.classifier = torch.nn.Linear(2 * bert_embedding_dim, num_classes)
        elif config['similarity'] == "MLP":
            #  $[p_u, q_i, p_u - q_i]$ where $a-b$ is the element-wise difference between the vectors.
            if "k" in config and config["k"] not in ['', 0]:
                in_size = 3 * config['k']
            else:
                in_size = 3 * bert_embedding_dim
            mlp_layers = []
            for out_size in config['MLP_layers']:
                mlp_layers.append(torch.nn.Linear(in_size, out_size))
                mlp_layers.append(torch.nn.Dropout(config['MLP_dropout']))
                if config['MLP_activation'] == "ReLU":
                    mlp_layers.append(torch.nn.ReLU())
                else:
                    raise ValueError("Not implemented!")
                in_size = out_size
            mlp_layers.append(torch.nn.Linear(in_size, num_classes))
            mlp_layers.append(torch.nn.Dropout(config['MLP_dropout']))
            if config['MLP_activation'] == "ReLU":
                mlp_layers.append(torch.nn.ReLU())
            else:
                raise ValueError("Not implemented!")
            self.mlp = torch.nn.Sequential(*mlp_layers)

        self.agg_strategy = config['agg_strategy']
        self.chunk_agg_strategy = config['chunk_agg_strategy']
        self.batch_size = config['precalc_batch_size']

        start = time.time()
        user_rep_file = f"{self.agg_strategy}_{self.chunk_agg_strategy}_" \
                        f"id{config['append_id']}_tb{config['tune_BERT']}_user_representation.pkl"
        if os.path.exists(os.path.join(prec_dir, user_rep_file)):
            weights = torch.load(os.path.join(prec_dir, user_rep_file), map_location=device)
        else:
            weights = self.create_representations(bert, bert_embeddings, user_info, padding_token, device,
                                                  config['append_id'], INTERNAL_USER_ID_FIELD,
                                                  self.user_id_embedding if config["append_id"] else None)
            torch.save(weights, open(os.path.join(prec_dir, user_rep_file)))
        self.user_rep = torch.nn.Embedding.from_pretrained(weights, freeze=self.freeze_prec_rep)  # todo freeze? or unfreeze?
        print(f"user rep loaded in {time.time()-start}")
        start = time.time()
        item_rep_file = f"{self.agg_strategy}_{self.chunk_agg_strategy}_" \
                        f"id{config['append_id']}_tb{config['tune_BERT']}_item_representation.pkl"
        if os.path.exists(os.path.join(prec_dir, item_rep_file)):
            weights = torch.load(os.path.join(prec_dir, item_rep_file), map_location=device)
        else:
            weights = self.create_representations(bert, bert_embeddings, item_info, padding_token, device,
                                                  config['append_id'], INTERNAL_ITEM_ID_FIELD,
                                                  self.item_id_embedding if config["append_id"] else None)
            torch.save(weights, open(os.path.join(prec_dir, item_rep_file)))
        self.item_rep = torch.nn.Embedding.from_pretrained(weights, freeze=self.freeze_prec_rep)  # todo freeze? or unfreeze?
        print(f"item rep loaded in {time.time()-start}")

    def create_representations(self, bert, bert_embeddings, info, padding_token, device,
                               append_id, id_field, id_embedding=None):
        collate_fn = CollateRepresentationBuilder(padding_token=padding_token)
        dataloader = DataLoader(info, batch_size=self.batch_size, collate_fn=collate_fn)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            # go over chunks:
            outputs = []
            ids = batch[id_field].to(device)
            for input_ids, att_mask in zip(batch['chunks_input_ids'], batch['chunks_attention_mask']):
                input_ids = input_ids.to(device)
                att_mask = att_mask.to(device)
                if append_id:
                    id_embeds = id_embedding(ids)
                    token_embeddings = bert_embeddings.forward(input_ids)
                    cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                    other_tokens = token_embeddings[:, 1:]
                    # insert user_id embedding after the especial CLS token:
                    concat_ids = torch.concat([torch.concat([cls_tokens, id_embeds], dim=1), other_tokens], dim=1)
                    concat_masks = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask], dim=1)
                    output = bert.forward(inputs_embeds=concat_ids,
                                          attention_mask=concat_masks)
                else:
                    output = bert.forward(input_ids=input_ids,
                                          attention_mask=att_mask)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            rep = torch.stack(outputs).max(dim=0).values
            reps.append(rep)
        return torch.concat(reps)

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        if hasattr(self, 'transform_u'):
            user_rep = self.transform_u(user_rep)
            item_rep = self.transform_i(item_rep)
        if hasattr(self, 'classifier'):
            result = self.classifier(torch.concat([user_rep, item_rep], dim=1))
        elif hasattr(self, 'bias'):
            result = torch.sum(torch.mul(user_rep, item_rep), dim=1)
            result = result + self.item_bias[item_ids] + self.user_bias[user_ids]
            result = result + self.bias
            result = result.unsqueeze(1)
        elif hasattr(self, 'mlp'):
            result = self.mlp(torch.concat([user_rep, item_rep, user_rep-item_rep], dim=1))
        return result  # do not apply sigmoid and use BCEWithLogitsLoss

