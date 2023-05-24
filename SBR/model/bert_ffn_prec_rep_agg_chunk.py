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
        self.append_embedding_ffn = model_config["append_embedding_ffn"] if "append_embedding_ffn" in model_config else False
        self.append_embedding_after_ffn = model_config["append_embedding_after_ffn"] if "append_embedding_after_ffn" in model_config else False

        if self.append_embedding_ffn and self.append_embedding_after_ffn:
            raise ValueError("only one of the append embeddings can be given")

        self.max_user_chunks = dataset_config['max_num_chunks_user']
        self.max_item_chunks = dataset_config['max_num_chunks_item']

        dim1user = dim1item = bert_embedding_dim

        if self.append_cf_after:
            CF_model_weights = torch.load(model_config['append_CF_after_model_path'], map_location="cpu")[
                'model_state_dict']
            # note: no need to match item and user ids only due to them being created with the same framework where we sort ids.
            # otherwise there needs to be a matching
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'], freeze=True)
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'], freeze=True)
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

        # TODO load reps

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        BERT_DIM = self.bert_embeddings.embedding_dim

        input_ids = batch['user_input_ids']
        att_mask = batch['user_attention_mask']

        # user_text = []
        # if DEBUG:
        #     for uid, ui in zip(user_ids, input_ids):
        #         user_text.append(" ".join(self.tokenizer.convert_ids_to_tokens(ui)))

        if self.use_cf or self.append_embedding_to_text:
            if self.use_cf:
                user_embed = self.user_embedding_CF(user_ids)
            elif self.append_embedding_to_text:
                user_embed = self.user_embedding(user_ids)
            user_embed = user_embed.to(self.device)
            if user_embed.shape[1] < BERT_DIM:
                user_embed = torch.concat([user_embed, torch.zeros((user_embed.shape[0], BERT_DIM-user_embed.shape[1]), device=self.device)], dim=1)
            user_embed = user_embed.unsqueeze(1)

            token_embeddings = self.bert_embeddings.forward(input_ids)
            cls_tokens = token_embeddings[:, 0].unsqueeze(1)
            other_tokens = token_embeddings[:, 1:]
            # insert cf embedding after the especial CLS token:
            concat_ids = torch.concat([cls_tokens, user_embed, other_tokens], dim=1)
            att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=self.device), att_mask], dim=1)
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
            summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
            user_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

        # append cf to the end of the ch reps :
        if self.append_cf_after:
            user_rep = torch.cat([user_rep, self.user_embedding_CF(user_ids)], dim=1)
        # id as single thing
        if self.append_id_ffn:
            user_rep = torch.cat([user_rep, self.user_ids_normalized(user_ids)], dim=1)
        # id as embedding to be learned:
        if self.append_embedding_ffn:
            user_rep = torch.cat([user_rep, self.user_embedding(user_ids)], dim=1)

        for k in range(len(self.user_linear_layers) - 1):
            user_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_rep))
        user_rep = self.user_linear_layers[-1](user_rep)

        if self.append_embedding_after_ffn:
            user_rep = torch.cat([user_rep, self.user_embedding(user_ids)], dim=1)

        input_ids = batch['item_input_ids']
        att_mask = batch['item_attention_mask']
        if self.use_cf or self.append_embedding_to_text:
            if self.use_cf:
                item_embed = self.item_embedding_CF(item_ids)
            elif self.append_embedding_to_text:
                item_embed = self.item_embedding(item_ids)
            item_embed = item_embed.to(self.device)
            if item_embed.shape[1] < BERT_DIM:
                item_embed = torch.concat(
                    [item_embed, torch.zeros((item_embed.shape[0], BERT_DIM - item_embed.shape[1]), device=self.device)], dim=1)
            item_embed = item_embed.unsqueeze(1)
            token_embeddings = self.bert_embeddings.forward(input_ids)
            cls_tokens = token_embeddings[:, 0].unsqueeze(1)
            other_tokens = token_embeddings[:, 1:]
            # insert cf embedding after the especial CLS token:
            concat_ids = torch.concat([cls_tokens, item_embed, other_tokens], dim=1)
            att_mask = torch.concat([torch.ones((input_ids.shape[0], 1), device=att_mask.device), att_mask], dim=1)
            # TODO segment encoding?
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

        # append cf to the end of the ch reps :
        if self.append_cf_after:  # did for tr as well before, not changed it to only for cases for ffn ... maybe another name is needed
            item_rep = torch.cat([item_rep, self.item_embedding_CF(item_ids)], dim=1)
        # id as single thing
        if self.append_id_ffn:
            item_rep = torch.cat([item_rep, self.item_ids_normalized(item_ids)], dim=1)
        # id as embedding to be learned:
        if self.append_embedding_ffn:
            item_rep = torch.cat([item_rep, self.item_embedding(item_ids)], dim=1)

        for k in range(len(self.item_linear_layers) - 1):
            item_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_rep))
        item_rep = self.item_linear_layers[-1](item_rep)

        if self.append_embedding_after_ffn:
            item_rep = torch.cat([item_rep, self.item_embedding(item_ids)], dim=1)

        result = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        result = result.unsqueeze(1)
        return result  #, user_text  # user_text for debug  # do not apply sigmoid here, later in the trainer if we wanted we would

