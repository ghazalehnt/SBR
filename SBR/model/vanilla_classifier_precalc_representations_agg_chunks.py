import os.path

import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(torch.nn.Module):
    def __init__(self, model_config, users, items, device, dataset_config,
                 use_ffn=False, use_item_bias=False, use_user_bias=False):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks, self).__init__()
        bert_embedding_dim = 768
        n_users = users.shape[0]
        n_items = items.shape[0]

        self.use_ffn = use_ffn
        self.use_item_bias = use_item_bias
        self.use_user_bias = use_user_bias

        if self.use_ffn:
            self.transform_u_1 = torch.nn.Linear(bert_embedding_dim, model_config['k1'])
            self.transform_i_1 = torch.nn.Linear(bert_embedding_dim, model_config['k1'])
            self.transform_u_2 = torch.nn.Linear(model_config['k1'], model_config['k2'])
            self.transform_i_2 = torch.nn.Linear(model_config['k1'], model_config['k2'])

        if self.use_item_bias:
            self.item_bias = torch.nn.Parameter(torch.zeros(n_items))

        if self.use_user_bias:
            self.user_bias = torch.nn.Parameter(torch.zeros(n_users))

        self.chunk_agg_strategy = model_config['chunk_agg_strategy']
        max_num_chunks_user = dataset_config['max_num_chunks_user']
        max_num_chunks_item = dataset_config['max_num_chunks_item']

        if "use_random_reps" in model_config and model_config["use_random_reps"] is True:
            self.chunk_user_reps = {}
            for c in range(max_num_chunks_user):
                self.chunk_user_reps[c] = torch.nn.Embedding(n_users, 768, device=device)
                self.chunk_user_reps[c].requires_grad_(False)

            self.chunk_item_reps = {}
            for c in range(max_num_chunks_item):
                self.chunk_item_reps[c] = torch.nn.Embedding(n_items, 768, device=device)
                self.chunk_item_reps[c].requires_grad_(False)
        else:
            prec_path = os.path.join(dataset_config['dataset_path'], 'precomputed_reps',
                                     f"size{dataset_config['user_chunk_size']}_"
                                     f"cs-{dataset_config['case_sensitive']}_"
                                     f"nn-{dataset_config['normalize_negation']}_"
                                     f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
            user_rep_file = f"user_representation_" \
                            f"{model_config['agg_strategy']}_" \
                            f"id{model_config['append_id']}_" \
                            f"tb{model_config['tune_BERT']}_" \
                            f"cf{model_config['use_CF']}_" \
                            f"{'-'.join(dataset_config['user_text'])}_" \
                            f"{dataset_config['user_item_text_choice']}_" \
                            f"{dataset_config['user_item_text_tie_breaker'] if dataset_config['user_text_filter'] in ['', 'item_sentence_SBERT'] else ''}_" \
                            f"{dataset_config['user_text_filter'] if len(dataset_config['user_text_filter']) > 0 else 'no-filter'}" \
                            f"{'_i' + '-'.join(dataset_config['item_text']) if dataset_config['user_text_filter'] in ['item_sentence_SBERT'] else ''}" \
                            f".pkl"
            if os.path.exists(os.path.join(prec_path, user_rep_file)):
                user_chunk_reps_dict = torch.load(os.path.join(prec_path, user_rep_file))
                user_chunk_reps = []
                cnt = 0
                for u in users.sort('user_id'):
                    ex_user_id = u["user_id"]
                    user_chunk_reps.append(user_chunk_reps_dict[ex_user_id])
                    if cnt != u[INTERNAL_USER_ID_FIELD]:
                        raise RuntimeError("wrong user!")
                    cnt += 1
            else:
                raise ValueError(
                    f"Precalculated user embedding does not exist! {os.path.join(prec_path, user_rep_file)}")
            prec_path = os.path.join(dataset_config['dataset_path'], 'precomputed_reps',
                                     f"size{dataset_config['item_chunk_size']}_"
                                     f"cs-{dataset_config['case_sensitive']}_"
                                     f"nn-{dataset_config['normalize_negation']}_"
                                     f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
            item_rep_file = f"item_representation_" \
                            f"{model_config['agg_strategy']}_" \
                            f"id{model_config['append_id']}_" \
                            f"tb{model_config['tune_BERT']}_" \
                            f"cf{model_config['use_CF']}_" \
                            f"{'-'.join(dataset_config['item_text'])}" \
                            f".pkl"
            if os.path.exists(os.path.join(prec_path, item_rep_file)):
                item_chunks_reps_dict = torch.load(os.path.join(prec_path, item_rep_file))
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
                    f"Precalculated item embedding does not exist! {os.path.join(prec_path, item_rep_file)}")

            self.chunk_user_reps = {}
            for c in range(max_num_chunks_user):
                ch_rep = []
                for user_chunks in user_chunk_reps:
                    if c < len(user_chunks):
                        ch_rep.append(user_chunks[c])
                    else:
                        if self.chunk_agg_strategy == "max_pool":
                            ch_rep.append(user_chunks[0])  # if user has fewer than c chunks, add its chunk0
                        else:
                            raise NotImplementedError()
                self.chunk_user_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=model_config['freeze_prec_reps']) # TODO? or concat? stach -> n*1*768 , concatn*768
                self.chunk_user_reps[c].to(device)

            self.chunk_item_reps = {}
            for c in range(max_num_chunks_item):
                ch_rep = []
                for item_chunks in item_chunks_reps:
                    if c < len(item_chunks):
                        ch_rep.append(item_chunks[c])
                    else:
                        if self.chunk_agg_strategy == "max_pool":
                            ch_rep.append(item_chunks[0])  # if item has fewer than c chunks, add its chunk0
                        else:
                            raise NotImplementedError()
                self.chunk_item_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=model_config['freeze_prec_reps'])  # TODO? or concat? stach -> n*1*768 , concatn*768
                self.chunk_item_reps[c].to(device)

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        # todo for users/items who had fewer than max-chunks, I put their chunk[0] for the missing chunks, so the output would be same az 0, max pooling gets chunk0
        user_reps = []
        for c in range(len(self.chunk_user_reps.keys())):
            user_ch_rep = self.chunk_user_reps[c](user_ids)
            if self.use_ffn:
                user_ch_rep = torch.nn.functional.relu(self.transform_u_1(user_ch_rep))
                user_ch_rep = self.transform_u_2(user_ch_rep)
            user_reps.append(user_ch_rep)
        user_reps = torch.stack(user_reps).max(dim=0).values

        item_reps = []
        for c in range(len(self.chunk_item_reps.keys())):
            item_ch_rep = self.chunk_item_reps[c](item_ids)
            if self.use_ffn:
                item_ch_rep = torch.nn.functional.relu(self.transform_i_1(item_ch_rep))
                item_ch_rep = self.transform_i_2(item_ch_rep)
            item_reps.append(item_ch_rep)
        item_reps = torch.stack(item_reps).max(dim=0).values

        result = torch.sum(torch.mul(user_reps, item_reps), dim=1)
        if self.use_item_bias:
            result = result + self.item_bias[item_ids]
        if self.use_user_bias:
            result = result + self.user_bias[user_ids]
        result = result.unsqueeze(1)
        return result  # do not apply sigmoid here, later in the trainer if we wanted we would

