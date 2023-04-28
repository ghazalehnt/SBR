import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD

# DEBUG = True

class BertFFNUserTextProfileItemTextProfileEndToEnd(torch.nn.Module):
    def __init__(self, model_config, device, dataset_config, users, items):
        super(BertFFNUserTextProfileItemTextProfileEndToEnd, self).__init__()
        bert_embedding_dim = 768
        self.append_cf_after = model_config["append_CF_after"] if "append_CF_after" in model_config else False
        self.agg_strategy = model_config['agg_strategy']
        self.device = device
        self.append_id_ffn = model_config['append_id_ffn']

        if dataset_config['max_num_chunks_user'] > 1 or dataset_config['max_num_chunks_item'] > 1:
            raise ValueError("max chunk should be set to 1 ")

        if self.append_cf_after:
            CF_model_weights = torch.load(model_config['append_CF_after_model_path'], map_location="cpu")[
                'model_state_dict']
            # note: no need to match item and user ids only due to them being created with the same framework where we sort ids.
            # otherwise there needs to be a matching
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'], freeze=True)
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'], freeze=True)

        dim1user = dim1item = bert_embedding_dim
        if self.append_cf_after:
            dim1user = bert_embedding_dim + self.user_embedding_CF.embedding_dim
            dim1item = bert_embedding_dim + self.item_embedding_CF.embedding_dim
        # adding id as integer TODO test
        if self.append_id_ffn:
            dim1user += 1
            dim1item += 1
            self.user_ids_normalized = torch.nn.Embedding.from_pretrained(((torch.tensor(users[INTERNAL_USER_ID_FIELD])+1) / len(users)).to(self.device).unsqueeze(1))
            self.item_ids_normalized = torch.nn.Embedding.from_pretrained(((torch.tensor(items[INTERNAL_ITEM_ID_FIELD])+1) / len(items)).to(self.device).unsqueeze(1))

        self.user_linear_layers = [torch.nn.Linear(dim1user, model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["user_k"])):
            self.user_linear_layers.append(torch.nn.Linear(model_config["user_k"][k-1], model_config["user_k"][k], device=self.device))

        self.item_linear_layers = [torch.nn.Linear(dim1item, model_config["item_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            self.item_linear_layers.append(torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))

        self.bert = transformers.AutoModel.from_pretrained(model_config['pretrained_model'])
        if model_config["tune_BERT"] is True:
            self.bert.requires_grad_(True)
            # freeze layers other than last one:
            freeze_modules = [self.bert.embeddings, self.bert.encoder.layer[:11]]
            for module in freeze_modules:
                for param in module.parameters():
                    param.requires_grad = False
        self.bert_embeddings = self.bert.get_input_embeddings()
        BERT_DIM = self.bert_embeddings.embedding_dim
        self.use_cf = model_config["use_CF"]
        if self.use_cf:
            CF_model_weights = torch.load(model_config['CF_model_path'], map_location="cpu")['model_state_dict']
            embedding_dim = CF_model_weights['user_embedding.weight'].shape[-1]
            if embedding_dim > BERT_DIM:
                raise ValueError("The CF embedding cannot be bigger than BERT dim")
            elif embedding_dim < BERT_DIM:
                print("CF embedding dim was smaller than BERT dim, therefore will be filled with  0's.")
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'])
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'])
        # # for debug:
        # if DEBUG:
        #     self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_config['pretrained_model'])

    def log(self, batch, which):
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']
        output = self.bert.forward(input_ids=input_ids,
                                   attention_mask=att_mask)
        if self.agg_strategy == "CLS":
            bert_rep = output.pooler_output
        elif self.agg_strategy == "mean_last":
            tokens_embeddings = output.last_hidden_state
            mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
            tokens_embeddings = tokens_embeddings * mask
            sum_tokons = torch.sum(tokens_embeddings, dim=1)
            summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
            bert_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

        if which == "user":
            ffn_rep = bert_rep.copy()
            for k in range(len(self.user_linear_layers)-1):
                ffn_rep = torch.nn.functional.relu(self.user_linear_layers[k](ffn_rep))
            ffn_rep = self.user_linear_layers[-1](ffn_rep)
        elif which == "item":
            ffn_rep = bert_rep.copy()
            for k in range(len(self.item_linear_layers) - 1):
                ffn_rep = torch.nn.functional.relu(self.item_linear_layers[k](ffn_rep))
            ffn_rep = self.item_linear_layers[-1](ffn_rep)
        else:
            raise ValueError("user or item?")
        return bert_rep, ffn_rep

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

        if self.use_cf is True:
            cf_embeds = self.user_embedding_CF(user_ids)
            cf_embeds = cf_embeds.to(self.device)
            if self.user_embedding_CF.embedding_dim < BERT_DIM:
                cf_embeds = torch.concat([cf_embeds, torch.zeros((cf_embeds.shape[0], BERT_DIM-cf_embeds.shape[1]), device=self.device)], dim=1)
            cf_embeds = cf_embeds.unsqueeze(1)
            token_embeddings = self.bert_embeddings.forward(input_ids)
            cls_tokens = token_embeddings[:, 0].unsqueeze(1)
            other_tokens = token_embeddings[:, 1:]
            # insert cf embedding after the especial CLS token:
            concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
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
        # id as single thing TODO test
        if self.append_id_ffn:
            user_rep = torch.cat([user_rep, self.user_ids_normalized(user_ids)], dim=1)
        for k in range(len(self.user_linear_layers) - 1):
            user_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_rep))
        user_rep = self.user_linear_layers[-1](user_rep)

        input_ids = batch['item_input_ids']
        att_mask = batch['item_attention_mask']
        if self.use_cf is True:
            cf_embeds = self.item_embedding_CF(item_ids)
            cf_embeds = cf_embeds.to(self.device)
            if self.item_embedding_CF.embedding_dim < BERT_DIM:
                cf_embeds = torch.concat([cf_embeds, torch.zeros((cf_embeds.shape[0], BERT_DIM - cf_embeds.shape[1]), device=cf_embeds.device)], dim=1)
            cf_embeds = cf_embeds.unsqueeze(1)
            token_embeddings = self.bert_embeddings.forward(input_ids)
            cls_tokens = token_embeddings[:, 0].unsqueeze(1)
            other_tokens = token_embeddings[:, 1:]
            # insert cf embedding after the especial CLS token:
            concat_ids = torch.concat([torch.concat([cls_tokens, cf_embeds], dim=1), other_tokens], dim=1)
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
        # id as single thing TODO test
        if self.append_id_ffn:
            item_rep = torch.cat([item_rep, self.item_ids_normalized(item_ids)], dim=1)

        for k in range(len(self.item_linear_layers) - 1):
            item_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_rep))
        item_rep = self.item_linear_layers[-1](item_rep)

        result = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        result = result.unsqueeze(1)
        return result  #, user_text  # user_text for debug  # do not apply sigmoid here, later in the trainer if we wanted we would

