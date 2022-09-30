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
    def __init__(self, config, n_users, n_items, num_classes, user_info, item_info, padding_token, device, prec_dir,
                 dataset_config):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculated, self).__init__()
        bert_embedding_dim = 768

        if "k" in config and config["k"] not in ['', 0]:
            self.transform_u = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.transform_i = torch.nn.Linear(bert_embedding_dim, config['k'])

        if "k1" in config and config["k1"] not in ['', 0]:
            self.transform_u_1 = torch.nn.Linear(bert_embedding_dim, config['k1'])
            self.transform_i_1 = torch.nn.Linear(bert_embedding_dim, config['k1'])
        if "k2" in config and config["k2"] not in ['', 0]:
            self.transform_u_2 = torch.nn.Linear(config['k1'], config['k2'])
            self.transform_i_2 = torch.nn.Linear(config['k1'], config['k2'])

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
                if config['MLP_activation'] == "ReLU":
                    mlp_layers.append(torch.nn.ReLU())
                else:
                    raise ValueError("Not implemented!")
                mlp_layers.append(torch.nn.Dropout(config['MLP_dropout']))
                in_size = out_size
            mlp_layers.append(torch.nn.Linear(in_size, num_classes))
            self.mlp = torch.nn.Sequential(*mlp_layers)
        else:
            raise ValueError("Not implemented")

        # it is better to always freeze the reps, and not append a random ID... so freeze_prec_reps:True, append_id:False
        freeze_prec_rep = config['freeze_prec_reps']
        agg_strategy = config['agg_strategy']
        chunk_agg_strategy = config['chunk_agg_strategy']

        user_rep_file = f"user_representation_" \
                        f"{agg_strategy}_{chunk_agg_strategy}_" \
                        f"id{config['append_id']}_" \
                        f"tb{config['tune_BERT']}_" \
                        f"cf{config['use_CF']}_" \
                        f"ch{dataset_config['max_num_chunks_user']}_" \
                        f"{'-'.join(dataset_config['user_text'])}_" \
                        f"{dataset_config['user_review_choice']}_" \
                        f"{dataset_config['review_tie_breaker'] if dataset_config['user_text_filter'] not in ['', 'item_sentence_SBERT'] else ''}_" \
                        f"{dataset_config['user_text_filter'] if len(dataset_config['user_text_filter']) > 0 else 'no-filter'}" \
                        f"{'_i' + '-'.join(config['dataset']['item_text']) if config['dataset']['user_text_filter'] in ['item_sentence_SBERT'] else ''}" \
                        f".pkl"
        if os.path.exists(os.path.join(prec_dir, user_rep_file)):
            weights = torch.load(os.path.join(prec_dir, user_rep_file), map_location=device)
        else:
            raise ValueError(f"Precalculated user embedding does not exist! {os.path.join(prec_dir, user_rep_file)}")
        self.user_rep = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_prec_rep)  # todo freeze? or unfreeze?

        item_rep_file = f"item_representation_" \
                        f"{agg_strategy}_{chunk_agg_strategy}_" \
                        f"id{config['append_id']}_" \
                        f"tb{config['tune_BERT']}_" \
                        f"cf{config['use_CF']}_" \
                        f"ch{dataset_config['max_num_chunks_item']}_" \
                        f"{'-'.join(dataset_config['item_text'])}" \
                        f".pkl"
        if os.path.exists(os.path.join(prec_dir, item_rep_file)):
            weights = torch.load(os.path.join(prec_dir, item_rep_file), map_location=device)
        else:
            raise ValueError(f"Precalculated item embedding does not exist! {os.path.join(prec_dir, item_rep_file)}")
        self.item_rep = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_prec_rep)  # todo freeze? or unfreeze?

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        if hasattr(self, 'transform_u'):
            user_rep = self.transform_u(user_rep)
            item_rep = self.transform_i(item_rep)
        elif hasattr(self, 'transform_u_1'):
            user_rep = torch.nn.functional.relu(self.transform_u_1(user_rep))
            item_rep = torch.nn.functional.relu(self.transform_i_1(item_rep))
            user_rep = self.transform_u_2(user_rep)
            item_rep = self.transform_i_2(item_rep)
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

