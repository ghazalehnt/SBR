import os

import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

# DEBUG = True


class BertFFNPrecomputedRepsChunkAgg(torch.nn.Module):
    def __init__(self, model_config, device, dataset_config, users, items):
        super(BertFFNPrecomputedRepsChunkAgg, self).__init__()
        if "tune_BERT" in model_config and model_config["tune_BERT"] is True:
            raise ValueError("tune_BERT must be False")
        if "use_CF" in model_config and model_config["use_CF"]:
            raise ValueError("use_CF cannot be true")
        if "append_embedding_to_text" in model_config and model_config["append_embedding_to_text"]:
            raise ValueError("append_embedding_to_text cannot be true")

        bert_embedding_dim = 768
        self.chunk_agg_strategy = model_config['chunk_agg_strategy']
        self.device = device
        self.append_cf_after = model_config["append_CF_after"] if "append_CF_after" in model_config else False
        self.append_cf_after_ffn = model_config["append_CF_after_ffn"] if "append_CF_after_ffn" in model_config else False
        self.append_embedding_ffn = model_config["append_embedding_ffn"] if "append_embedding_ffn" in model_config else False
        self.append_embedding_after_ffn = model_config["append_embedding_after_ffn"] if "append_embedding_after_ffn" in model_config else False

        if self.append_embedding_ffn and self.append_embedding_after_ffn:
            raise ValueError("only one of the append embeddings can be given")

        self.max_user_chunks = dataset_config['max_num_chunks_user']
        self.max_item_chunks = dataset_config['max_num_chunks_item']

        dim1user = dim1item = bert_embedding_dim

        if self.append_cf_after or self.append_cf_after_ffn:
            CF_model_weights = torch.load(model_config['append_CF_after_model_path'], map_location="cpu")[
                'model_state_dict']
            # note: no need to match item and user ids only due to them being created with the same framework where we sort ids.
            # otherwise there needs to be a matching
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'],
                                                                        freeze=model_config["freeze_CF"] if "freeze_CF" in model_config else True)
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'],
                                                                        freeze=model_config["freeze_CF"] if "freeze_CF" in model_config else True)
            if self.append_cf_after:
                dim1user += self.user_embedding_CF.embedding_dim
                dim1item += self.item_embedding_CF.embedding_dim

        if self.append_embedding_ffn or self.append_embedding_after_ffn:
            if self.append_embedding_ffn:
                dim1user += model_config["user_embedding"]
                dim1item += model_config["item_embedding"]

            self.user_embedding = torch.nn.Embedding(len(users), model_config["user_embedding"])
            self.item_embedding = torch.nn.Embedding(len(items), model_config["item_embedding"])
            if "embed_init" in model_config:
                if model_config["embed_init"] == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.user_embedding.weight)
                    torch.nn.init.xavier_uniform_(self.item_embedding.weight)
                elif model_config["embed_init"] == "xavier_normal":
                    torch.nn.init.xavier_normal_(self.user_embedding.weight)
                    torch.nn.init.xavier_normal_(self.item_embedding.weight)
                else:
                    raise NotImplementedError("embed init not implemented")

        user_layers = [torch.nn.Linear(dim1user, model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["user_k"])):
            user_layers.append(torch.nn.Linear(model_config["user_k"][k-1], model_config["user_k"][k], device=self.device))
        self.user_linear_layers = torch.nn.ModuleList(user_layers)

        item_layers = [torch.nn.Linear(dim1item, model_config["item_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            item_layers.append(torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))
        self.item_linear_layers = torch.nn.ModuleList(item_layers)

        cf_emb_dim = ""
        if model_config['use_CF']:
            cf_emb_dim = f"_MF-{model_config['CF_embedding_dim']}"

        user_prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
                                      f"size{dataset_config['user_chunk_size']}_"
                                      f"cs-{dataset_config['case_sensitive']}_"
                                      f"nn-{dataset_config['normalize_negation']}_"
                                      f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
        user_rep_file = f"user_representation_" \
                        f"{model_config['agg_strategy']}_" \
                        f"id{model_config['append_id']}_" \
                        f"cf{model_config['use_CF']}_" \
                        f"{dataset_config['user_text_file_name']}" \
                        f".pkl"
        if os.path.exists(os.path.join(user_prec_path, user_rep_file)):
            user_chunk_reps_dict = torch.load(os.path.join(user_prec_path, user_rep_file), map_location=torch.device('cpu'))
            user_chunks_reps = []
            cnt = 0
            for u in users.sort('user_id'):
                ex_user_id = u["user_id"]
                user_chunks_reps.append(user_chunk_reps_dict[ex_user_id])
                if cnt != u[INTERNAL_USER_ID_FIELD]:
                    raise RuntimeError("wrong user!")
                cnt += 1
        else:
            raise ValueError(
                f"Precalculated user embedding does not exist! {os.path.join(user_prec_path, user_rep_file)}")

        item_prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
                                      f"size{dataset_config['item_chunk_size']}_"
                                      f"cs-{dataset_config['case_sensitive']}_"
                                      f"nn-{dataset_config['normalize_negation']}_"
                                      f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
        item_rep_file = f"item_representation_" \
                        f"{model_config['agg_strategy']}_" \
                        f"id{model_config['append_id']}_" \
                        f"cf{model_config['use_CF']}_" \
                        f"{dataset_config['item_text_file_name']}" \
                        f".pkl"
        if os.path.exists(os.path.join(item_prec_path, item_rep_file)):
            item_chunks_reps_dict = torch.load(os.path.join(item_prec_path, item_rep_file), map_location=torch.device('cpu'))
            item_chunks_reps = []
            cnt = 0
            for i in items.sort('item_id'):
                ex_item_id = i["item_id"]
                item_chunks_reps.append(item_chunks_reps_dict[ex_item_id])
                if cnt != i[INTERNAL_ITEM_ID_FIELD]:
                    raise RuntimeError("item user!")
                cnt += 1
        else:
            raise ValueError(
                f"Precalculated item embedding does not exist! {os.path.join(item_prec_path, item_rep_file)}")

        n_users = users.shape[0]
        n_items = items.shape[0]

        self.chunk_user_reps = {}
        self.user_chunk_masks = torch.zeros((n_users, self.max_user_chunks))
        for c in range(self.max_user_chunks):
            ch_rep = []
            for i, user_chunks in zip(range(n_users), user_chunks_reps):
                if c < len(user_chunks):
                    ch_rep.append(user_chunks[c])
                    self.user_chunk_masks[i][c] = 1
                else:
                    ch_rep.append(user_chunks[0])  # if user has fewer than c chunks, add its chunk0
            self.chunk_user_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep),
                                                                         freeze=True)

        self.chunk_item_reps = {}
        self.item_chunk_masks = torch.zeros((n_items, self.max_item_chunks))
        for c in range(self.max_item_chunks):
            ch_rep = []
            for i, item_chunks in zip(range(n_items), item_chunks_reps):
                if c < len(item_chunks):
                    ch_rep.append(item_chunks[c])
                    self.item_chunk_masks[i][c] = 1
                else:
                    ch_rep.append(item_chunks[0])  # if item has fewer than c chunks, add its chunk0
            self.chunk_item_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep),
                                                                         freeze=True)

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        # user
        chunk_device = self.chunk_user_reps[0].weight.device
        if chunk_device.__str__() == 'cpu':
            id_temp = user_ids.cpu()
        user_reps = []
        for c in range(len(self.chunk_user_reps.keys())):
            if chunk_device.__str__() == 'cpu':
                user_ch_rep = self.chunk_user_reps[c](id_temp)
                user_ch_rep = user_ch_rep.to(self.device)
            else:
                user_ch_rep = self.chunk_user_reps[c](user_ids)
            # append cf to the end of the ch reps :
            if self.append_cf_after:
                user_ch_rep = torch.cat([user_ch_rep, self.user_embedding_CF(user_ids)], dim=1)
            # id as embedding to be learned:
            if self.append_embedding_ffn:
                user_ch_rep = torch.cat([user_ch_rep, self.user_embedding(user_ids)], dim=1)

            for k in range(len(self.user_linear_layers) - 1):
                user_ch_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_ch_rep))
            user_ch_rep = self.user_linear_layers[-1](user_ch_rep)
            user_reps.append(user_ch_rep)

        if self.chunk_agg_strategy == "max_pool":
            user_rep = torch.stack(user_reps).max(dim=0).values
        elif self.chunk_agg_strategy == "avg":
            user_rep = torch.stack(user_reps).mean(dim=0)

        if self.append_embedding_after_ffn:
            user_rep = torch.cat([user_rep, self.user_embedding(user_ids)], dim=1)
        if self.append_cf_after_ffn:
            user_rep = torch.cat([user_rep, self.user_embedding_CF(user_ids)], dim=1)

        # item
        if chunk_device.__str__() == 'cpu':
            id_temp = item_ids.cpu()
        item_reps = []
        for c in range(len(self.chunk_item_reps.keys())):
            if chunk_device.__str__() == 'cpu':
                item_ch_rep = self.chunk_item_reps[c](id_temp)
                item_ch_rep = item_ch_rep.to(self.device)
            else:
                item_ch_rep = self.chunk_item_reps[c](item_ids)

            # append cf to the end of the ch reps :
            if self.append_cf_after:
                item_ch_rep = torch.cat([item_ch_rep, self.item_embedding_CF(item_ids)], dim=1)
            # id as embedding to be learned:
            if self.append_embedding_ffn:
                item_ch_rep = torch.cat([item_ch_rep, self.item_embedding(item_ids)], dim=1)

            for k in range(len(self.item_linear_layers) - 1):
                item_ch_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_ch_rep))
            item_ch_rep = self.item_linear_layers[-1](item_ch_rep)
            item_reps.append(item_ch_rep)

        if self.chunk_agg_strategy == "max_pool":
            item_rep = torch.stack(item_reps).max(dim=0).values
        elif self.chunk_agg_strategy == "avg":
            item_rep = torch.stack(item_reps).mean(dim=0)

        if self.append_embedding_after_ffn:
            item_rep = torch.cat([item_rep, self.item_embedding(item_ids)], dim=1)
        if self.append_cf_after_ffn:
            item_rep = torch.cat([item_rep, self.item_embedding_CF(item_ids)], dim=1)

        result = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        result = result.unsqueeze(1)
        return result  # do not apply sigmoid here, later in the trainer if we wanted we would

