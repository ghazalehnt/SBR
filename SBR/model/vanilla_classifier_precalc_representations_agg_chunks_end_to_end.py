import os.path

import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunksEndToEnd(torch.nn.Module):
    def __init__(self, model_config, users, items, device, dataset_config,
                 use_ffn=False, use_transformer=False, use_item_bias=False, use_user_bias=False):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunksEndToEnd, self).__init__()
        bert_embedding_dim = 768
        n_users = users.shape[0]
        n_items = items.shape[0]

        self.use_ffn = use_ffn
        self.use_transformer = use_transformer
        self.use_item_bias = use_item_bias
        self.use_user_bias = use_user_bias
        self.append_cf_after = model_config["append_CF_after"] if "append_CF_after" in model_config else False
        self.agg_strategy = model_config['agg_strategy']

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

        self.max_num_chunks_user = dataset_config['max_num_chunks_user']
        self.max_num_chunks_item = dataset_config['max_num_chunks_item']
        # if max_num_chunks_user > 1 or max_num_chunks_item > 1:
        #     raise ValueError("max chunk should be set to 1 ")  # TODO test this though
        # end-to-end
        self.bert = transformers.AutoModel.from_pretrained(model_config['pretrained_model'])
        if model_config["tune_BERT"] is True:
            self.bert.trainable = True
            self.bert.requires_grad_(True)
            for param in self.bert.parameters():
                param.requires_grad = True
        self.bert_embeddings = self.bert.get_input_embeddings()
        BERT_DIM = self.bert_embeddings.embedding_dim
        self.use_cf = model_config["use_CF"]
        if self.use_cf:
            # TODO segment encoding
            # self.bert_seg_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=768)
            CF_model_weights = torch.load(model_config['CF_model_path'], map_location=device)['model_state_dict']
            embedding_dim = CF_model_weights['user_embedding.weight'].shape[-1]
            if embedding_dim > BERT_DIM:
                raise ValueError("The CF embedding cannot be bigger than BERT dim")
            elif embedding_dim < BERT_DIM:
                print("CF embedding dim was smaller than BERT dim, therefore will be filled with  0's.")
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'])
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'])

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        # TODO try with max chunk > 1
        # TODO 1: try with item and user input to single bert
        # this for is on
        BERT_DIM = self.bert_embeddings.embedding_dim
        user_reps = []
        for c in range(self.max_num_chunks_user):
            input_ids = batch['user_chunks_input_ids'][c]
            att_mask = batch['user_chunks_attention_mask'][c]
            if self.use_cf is True:
                cf_embeds = self.user_embedding_CF(user_ids)
                if self.user_embedding_CF.embedding_dim < BERT_DIM:
                    cf_embeds = torch.concat([cf_embeds, torch.zeros((cf_embeds.shape[0], BERT_DIM-cf_embeds.shape[1]), device=cf_embeds.device)], dim=1)
                cf_embeds = cf_embeds.unsqueeze(1)
                token_embeddings = self.bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert cf embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask], dim=1)
                # TODO segment encoding
                output_u = self.bert.forward(inputs_embeds=concat_ids,
                                             attention_mask=att_mask)
            else:
                output_u = self.bert.forward(input_ids=input_ids,
                                             attention_mask=att_mask)
            if self.agg_strategy == "CLS":
                user_rep = output_u.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output_u.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)  #-> I see the point, but it's better to leave it as is to find the errors as in our case there should be something
                user_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

            if self.use_ffn:
                # append cf to the end of the ch reps :
                if self.append_cf_after:
                    user_rep = torch.cat([user_rep, self.user_embedding_CF(user_ids)], dim=1)
                user_rep = torch.nn.functional.relu(self.linear_u_1(user_rep))
                user_rep = self.linear_u_2(user_rep)
            user_reps.append(user_rep)
        if self.chunk_agg_strategy == "max_pool" and self.use_ffn:
            user_reps = torch.stack(user_reps).max(dim=0).values

        item_reps = []
        for c in range(self.max_num_chunks_item):
            input_ids = batch['item_chunks_input_ids'][c]
            att_mask = batch['item_chunks_attention_mask'][c]
            if self.use_cf is True:
                cf_embeds = self.item_embedding_CF(item_ids)
                if self.item_embedding_CF.embedding_dim < BERT_DIM:
                    cf_embeds = torch.concat([cf_embeds, torch.zeros((cf_embeds.shape[0], BERT_DIM - cf_embeds.shape[1]), device=cf_embeds.device)], dim=1)
                cf_embeds = cf_embeds.unsqueeze(1)
                token_embeddings = self.bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert cf embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
                att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask], dim=1)
                # TODO segment encoding
                output_i = self.bert.forward(inputs_embeds=concat_ids,
                                             attention_mask=att_mask)
            else:
                output_i = self.bert.forward(input_ids=input_ids,
                                             attention_mask=att_mask)
            if self.agg_strategy == "CLS":
                item_rep = output_i.pooler_output
            elif self.agg_strategy == "mean_last":
                tokens_embeddings = output_i.last_hidden_state
                mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
                tokens_embeddings = tokens_embeddings * mask
                sum_tokons = torch.sum(tokens_embeddings, dim=1)
                summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)  #-> I see the point, but it's better to leave it as is to find the errors as in our case there should be something
                item_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

            if self.use_ffn:
                # append cf to the end of the ch reps :
                if self.append_cf_after:  # did for tr as well before, not changed it to only for cases for ffn ... maybe another name is needed
                    item_rep = torch.cat([item_rep, self.item_embedding_CF(item_ids)], dim=1)
                item_rep = torch.nn.functional.relu(self.linear_i_1(item_rep))
                item_rep = self.linear_i_2(item_rep)

            item_reps.append(item_rep)
        if self.chunk_agg_strategy == "max_pool" and self.use_ffn:
            item_reps = torch.stack(item_reps).max(dim=0).values

        # TODO clean the code, similarity for ffn ones are not it used for example...
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

