import os.path

import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfileJointEndToEnd(torch.nn.Module):
    def __init__(self, model_config, users, items, device, dataset_config):
        super(VanillaClassifierUserTextProfileItemTextProfileJointEndToEnd, self).__init__()
        bert_embedding_dim = 768
        n_users = users.shape[0]
        n_items = items.shape[0]
        self.agg_strategy = model_config['agg_strategy']
        self.device = device

        if dataset_config['max_num_chunks_user'] > 1 or dataset_config['max_num_chunks_item'] > 1:
            raise ValueError("max chunk should be set to 1 ")

        # end-to-end
        bert_embedding_dim = 768
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
            CF_model_weights = torch.load(model_config['CF_model_path'], map_location=device)['model_state_dict']
            embedding_dim = CF_model_weights['user_embedding.weight'].shape[-1]
            if embedding_dim > BERT_DIM:
                raise ValueError("The CF embedding cannot be bigger than BERT dim")
            elif embedding_dim < BERT_DIM:
                print("CF embedding dim was smaller than BERT dim, therefore will be filled with  0's.")
            self.user_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['user_embedding.weight'])
            self.item_embedding_CF = torch.nn.Embedding.from_pretrained(CF_model_weights['item_embedding.weight'])

        self.segment_embedding = torch.nn.Embedding(num_embeddings=2, embedding_dim=bert_embedding_dim)  # 2 or more?
        self.classifier = torch.nn.Linear(bert_embedding_dim, 1)

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        BERT_DIM = self.bert_embeddings.embedding_dim
        input_ids = batch['input_ids']
        att_mask = batch['attention_mask']

        if self.use_cf is True:
            raise NotImplementedError("this part should be checked and changed!")
        else:
            output = self.bert.forward(input_ids=input_ids,
                                       attention_mask=att_mask)

        if self.agg_strategy == "CLS":
            user_item_rep = output.pooler_output
        elif self.agg_strategy == "mean_last":
            tokens_embeddings = output.last_hidden_state
            mask = att_mask.unsqueeze(-1).expand(tokens_embeddings.size()).float()
            tokens_embeddings = tokens_embeddings * mask
            sum_tokons = torch.sum(tokens_embeddings, dim=1)
            summed_mask = torch.clamp(att_mask.sum(1).type(torch.float), min=1e-9)
            user_item_rep = (sum_tokons.T / summed_mask).T # divide by how many tokens (1s) are in the att_mask

        result = self.classifier(user_item_rep)
        return result  # do not apply sigmoid here, later in the trainer if we wanted we would

