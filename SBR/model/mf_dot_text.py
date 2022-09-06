import torch

import os

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class MatrixFactorizatoinTextDotProduct(torch.nn.Module):
    def __init__(self, config, n_users, n_items, device, prec_dir, dataset_config):
        super(MatrixFactorizatoinTextDotProduct, self).__init__()

        self.user_embedding = torch.nn.Embedding(n_users, config["embedding_dim"])
        self.item_embedding = torch.nn.Embedding(n_items, config["embedding_dim"])

        self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
        self.item_bias = torch.nn.Parameter(torch.zeros(n_items))
        self.bias = torch.nn.Parameter(torch.zeros(1))

        bert_embedding_dim = 768
        if "k" in config and config["k"] not in ['', 0]:
            self.transform_u_text = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.transform_i_text = torch.nn.Linear(bert_embedding_dim, config['k'])
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
                        f"{dataset_config['review_tie_breaker'] if len(dataset_config['user_text_filter']) == 0 else ''}_" \
                        f"{dataset_config['user_text_filter'] if len(dataset_config['user_text_filter']) > 0 else 'no-filter'}" \
                        f".pkl"
        if os.path.exists(os.path.join(prec_dir, user_rep_file)):
            weights = torch.load(os.path.join(prec_dir, user_rep_file), map_location=device)
        else:
            raise ValueError(f"Precalculated user embedding does not exist! {os.path.join(prec_dir, user_rep_file)}")
        self.user_text_rep = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_prec_rep)

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
        self.item_text_rep = torch.nn.Embedding.from_pretrained(weights, freeze=freeze_prec_rep)

    def forward(self, batch):
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze()
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze()

        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)

        user_t_rep = self.user_text_rep(user_ids)
        item_t_rep = self.item_text_rep(item_ids)
        if hasattr(self, 'transform_u_text'):
            user_t_rep = self.transform_u_text(user_t_rep)
            item_t_rep = self.transform_i_text(item_t_rep)

        user_rep = torch.concat([user_embeds, user_t_rep], dim=1)
        item_rep = torch.concat([item_embeds, item_t_rep], dim=1)

        output = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        output = output + self.item_bias[item_ids] + self.user_bias[user_ids]
        output = output + self.bias

        return output.unsqueeze(1)  # do not apply sigmoid and use BCEWithLogitsLoss
