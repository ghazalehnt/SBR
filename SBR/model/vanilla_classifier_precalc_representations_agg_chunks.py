import os.path

import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(torch.nn.Module):
    def __init__(self, model_config, users, items, device, dataset_config,
                 use_ffn=False, use_transformer=False, use_item_bias=False, use_user_bias=False):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks, self).__init__()
        bert_embedding_dim = 768
        n_users = users.shape[0]
        n_items = items.shape[0]

        self.use_ffn = use_ffn
        self.use_transformer = use_transformer
        self.use_item_bias = use_item_bias
        self.use_user_bias = use_user_bias
        self.append_cf_after = model_config["append_CF_after"] if "append_CF_after" in model_config else False

        self.device = device

        # either ffn or transformer:
        if self.use_ffn and self.use_transformer:
            raise ValueError("Error: use either ffn or transformer not both!")

        # append_cf_after is for when we have the ffn layer, appending the cf to the BERT output and feeding it to ffn layer
        # so when it is True, the use_ffn layer must be True as well.
        if (self.use_ffn is False and self.use_transformer is False) and self.append_cf_after is True:
            raise ValueError("Error: config['append_CF_after'] is true, while use_ffn and use_transformer are both false!")

        if self.append_cf_after:
            CF_model_weights = torch.load(model_config['append_CF_after_model_path'], map_location=device)[
                'model_state_dict']
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'], freeze=True)
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'], freeze=True)

        if self.use_ffn:
            dim1user = dim1item = bert_embedding_dim
            if self.append_cf_after:
                dim1user = bert_embedding_dim + self.user_embedding_CF.embedding_dim
                dim1item = bert_embedding_dim + self.item_embedding_CF.embedding_dim
            self.linear_u_1 = torch.nn.Linear(dim1user, model_config['k1'])
            self.linear_u_2 = torch.nn.Linear(model_config['k1'], model_config['k2'])
            self.linear_i_1 = torch.nn.Linear(dim1item, model_config['k1'])
            self.linear_i_2 = torch.nn.Linear(model_config['k1'], model_config['k2'])

        # TODO 2 sep trans:
        # if self.use_transformer:
        #     dim_user = dim_item = bert_embedding_dim
        #     if self.append_cf_after: # TODO 2 ways: combine chunks and cf each as a token, or cat cf to chunks and pass this into transformer
        #         dim_user = bert_embedding_dim + self.user_embedding_CF.embedding_dim
        #         dim_item = bert_embedding_dim + self.item_embedding_CF.embedding_dim
        #     uencoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim_user, nhead=8, batch_first=True)
        #     self.transformer_encoder_user = torch.nn.TransformerEncoder(uencoder_layer, num_layers=6)
        #     iencoder_layer = torch.nn.TransformerEncoderLayer(d_model=dim_item, nhead=8, batch_first=True)
        #     self.transformer_encoder_item = torch.nn.TransformerEncoder(iencoder_layer, num_layers=6)

        # TODO single trans:
        if self.use_transformer:
            trans_dim = bert_embedding_dim
            ## this was with cat cf to each chunk embedding:
            # if self.append_cf_after: # TODO 2 ways: combine chunks and cf each as a token, or cat cf to chunks and pass this into transformer
            #     trans_dim = bert_embedding_dim + max(self.user_embedding_CF.embedding_dim, self.item_embedding_CF.embedding_dim)
            # self.segment_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=trans_dim)
            ## this is with having cf as input
            self.segment_embedding = torch.nn.Embedding(num_embeddings=4 if self.append_cf_after else 2, embedding_dim=trans_dim)
            self.CLS = torch.rand([1, trans_dim])
            self.CLS.requires_grad = True
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=trans_dim, nhead=8, batch_first=True)
            self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=6)

            self.classifier = torch.nn.Linear(trans_dim, 1)

        if self.use_item_bias:
            self.item_bias = torch.nn.Parameter(torch.zeros(n_items))

        if self.use_user_bias:
            self.user_bias = torch.nn.Parameter(torch.zeros(n_users))

        self.chunk_agg_strategy = model_config['chunk_agg_strategy']
        max_num_chunks_user = dataset_config['max_num_chunks_user']
        max_num_chunks_item = dataset_config['max_num_chunks_item']

        # if "use_random_reps" in model_config and model_config["use_random_reps"] is True:
        #     self.chunk_user_reps = {}
        #     for c in range(max_num_chunks_user):
        #         self.chunk_user_reps[c] = torch.nn.Embedding(n_users, 768, device=device)
        #         self.chunk_user_reps[c].requires_grad_(False)
        #
        #     self.chunk_item_reps = {}
        #     for c in range(max_num_chunks_item):
        #         self.chunk_item_reps[c] = torch.nn.Embedding(n_items, 768, device=device)
        #         self.chunk_item_reps[c].requires_grad_(False)
        # else:
        cf_emb_dim = ""
        if model_config["use_CF"]:
            cf_emb_dim = f"_MF-{model_config['CF_embedding_dim']}"

        prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
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
                        f"{dataset_config['user_item_text_choice'] if dataset_config['user_text_filter'] in ['', 'item_sentence_SBERT', 'item_per_chunk'] else ''}_" \
                        f"{dataset_config['user_item_text_tie_breaker'] if dataset_config['user_text_filter'] in ['', 'item_sentence_SBERT', 'item_per_chunk'] else ''}_" \
                        f"{dataset_config['user_text_filter'] if len(dataset_config['user_text_filter']) > 0 else 'no-filter'}" \
                        f"{'_i' + '-'.join(dataset_config['item_text']) if dataset_config['user_text_filter'] in ['item_sentence_SBERT'] else ''}" \
                        f".pkl"
        if os.path.exists(os.path.join(prec_path, user_rep_file)):
            user_chunk_reps_dict = torch.load(os.path.join(prec_path, user_rep_file), map_location=torch.device('cpu'))
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
        prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
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
            item_chunks_reps_dict = torch.load(os.path.join(prec_path, item_rep_file), map_location=torch.device('cpu'))
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
            self.chunk_user_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=model_config['freeze_prec_reps'])
#            self.chunk_user_reps[c].to(device)

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
            self.chunk_item_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=model_config['freeze_prec_reps'])
#            self.chunk_item_reps[c].to(device)

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        # todo for users/items who had fewer than max-chunks, I put their chunk[0] for the missing chunks, so the output would be same az 0, max pooling gets chunk0
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
            if self.use_ffn:
                # append cf to the end of the ch reps :
                if self.append_cf_after:
                    user_ch_rep = torch.cat([user_ch_rep, self.user_embedding_CF(user_ids)], dim=1)
                user_ch_rep = torch.nn.functional.relu(self.linear_u_1(user_ch_rep))
                user_ch_rep = self.linear_u_2(user_ch_rep)
            user_reps.append(user_ch_rep)
        if self.chunk_agg_strategy == "max_pool" and self.use_ffn:
            user_reps = torch.stack(user_reps).max(dim=0).values
        
        if chunk_device.__str__() == 'cpu':
            id_temp = item_ids.cpu()
        item_reps = []
        for c in range(len(self.chunk_item_reps.keys())):
            if chunk_device.__str__() == 'cpu':
                item_ch_rep = self.chunk_item_reps[c](id_temp)
                item_ch_rep = item_ch_rep.to(self.device)
            else:
                item_ch_rep = self.chunk_item_reps[c](item_ids)          

            if self.use_ffn:
                # append cf to the end of the ch reps :
                if self.append_cf_after:  # did for tr as well before, not changed it to only for cases for ffn ... maybe another name is needed
                    item_ch_rep = torch.cat([item_ch_rep, self.item_embedding_CF(item_ids)], dim=1)
                item_ch_rep = torch.nn.functional.relu(self.linear_i_1(item_ch_rep))
                item_ch_rep = self.linear_i_2(item_ch_rep)
            item_reps.append(item_ch_rep)
        if self.chunk_agg_strategy == "max_pool" and self.use_ffn:
            item_reps = torch.stack(item_reps).max(dim=0).values  # TODO or stack dim1 and max dim1

        # 2 sep trans
        # if self.use_transformer:
        #     item_reps = torch.stack(item_reps, dim=1)
        #     item_reps = self.transformer_encoder_item(item_reps)
        #     user_reps = torch.stack(user_reps, dim=1)
        #     user_reps = self.transformer_encoder_item(user_reps)
        #     if self.chunk_agg_strategy == "max_pool":
        #         item_reps = item_reps.max(dim=1).values
        #         user_reps = user_reps.max(dim=1).values

        # TODO clean the code, similarity for ffn ones are not used for example...
        if self.use_transformer:
            user_rep = torch.stack(user_reps, dim=1)
            item_rep = torch.stack(item_reps, dim=1)
            cls = self.CLS.repeat(user_ids.shape[0], 1, 1)
            cls = cls.to(self.device)
            if self.append_cf_after:
                user_cf = torch.zeros(cls.shape, device=self.device)
                user_cf[:, :, :self.user_embedding_CF.embedding_dim] = self.user_embedding_CF(user_ids).unsqueeze(dim=1)
                item_cf = torch.zeros(cls.shape, device=self.device)
                item_cf[:, :, :self.item_embedding_CF.embedding_dim] = self.item_embedding_CF(item_ids).unsqueeze(dim=1)
                input_seq = torch.concat([cls, user_rep, user_cf, item_rep, item_cf], dim=1)
                seg_seq = self.segment_embedding(torch.LongTensor([[0] + [0] * user_rep.shape[1] + [1] + [2] * item_rep.shape[1] + [3]] * item_ids.shape[0]).to(self.device))
            else:
                input_seq = torch.concat([cls, user_rep, item_rep], dim=1)
                seg_seq = self.segment_embedding(torch.LongTensor([[0] + [0] * user_rep.shape[1] + [1] * item_rep.shape[1]] * item_ids.shape[0]).to(self.device))
            # when appending cf to chunk:
            # seg_seq = self.segment_embedding(torch.LongTensor([[0] + [0] * user_rep.shape[1] + [1] * item_rep.shape[1]] * item_ids.shape[0]).to(self.device))

            output_seq = self.transformer_encoder(torch.add(input_seq, seg_seq))
            cls_out = output_seq[:, 0, :]
            result = self.classifier(cls_out)
        else:
            result = torch.sum(torch.mul(user_reps, item_reps), dim=1)
            if self.use_item_bias:
                result = result + self.item_bias[item_ids]
            if self.use_user_bias:
                result = result + self.user_bias[user_ids]
            result = result.unsqueeze(1)
        return result  # do not apply sigmoid here, later in the trainer if we wanted we would

