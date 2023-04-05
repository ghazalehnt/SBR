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
        c = 0  # TODO ONLY HAVE ONE CHUNK ANYWAYS, won't work with more

        user_input_ids = batch['user_chunks_input_ids'][c]
        user_att_mask = batch['user_chunks_attention_mask'][c]
        item_input_ids = batch['item_chunks_input_ids'][c]
        item_att_mask = batch['item_chunks_attention_mask'][c]

        if self.use_cf is True:
            user_cf_embeds = self.user_embedding_CF(user_ids)
            if self.user_embedding_CF.embedding_dim < BERT_DIM:
                user_cf_embeds = torch.concat([user_cf_embeds, torch.zeros((user_cf_embeds.shape[0], BERT_DIM-user_cf_embeds.shape[1]), device=user_cf_embeds.device)], dim=1)
            user_cf_embeds = user_cf_embeds.unsqueeze(1)
            user_token_embeddings = self.bert_embeddings.forward(user_input_ids)
            cls_tokens = user_token_embeddings[:, 0].unsqueeze(1)
            user_other_tokens = user_token_embeddings[:, 1:]
            # insert cf embedding after the especial CLS token:
            user_concat_ids = torch.concat([torch.concat([cls_tokens, user_cf_embeds], dim=1), user_other_tokens], dim=1)
            user_att_mask = torch.concat([torch.ones((user_input_ids.shape[0], 1), device=user_att_mask.device), user_att_mask], dim=1)

            item_cf_embeds = self.item_embedding_CF(item_ids)
            if self.item_embedding_CF.embedding_dim < BERT_DIM:
                item_cf_embeds = torch.concat([item_cf_embeds, torch.zeros((item_cf_embeds.shape[0], BERT_DIM - item_cf_embeds.shape[1]), device=item_cf_embeds.device)], dim=1)
            item_cf_embeds = item_cf_embeds.unsqueeze(1)
            item_token_embeddings = self.bert_embeddings.forward(item_input_ids)
            item_other_tokens = item_token_embeddings[:, 1:]
            item_concat_ids = torch.concat([item_cf_embeds, item_other_tokens], dim=1)
            item_att_mask = torch.concat([torch.ones((item_input_ids.shape[0], 1), device=item_att_mask.device), item_att_mask], dim=1)

            sep_tokens = self.bert_embeddings.forward(torch.tensor([102], device=self.device)).repeat(user_ids.shape[0], 1, 1)

            concat_ids = torch.concat([user_concat_ids, sep_tokens, item_concat_ids], dim=1)
            att_mask = torch.concat([user_att_mask, item_att_mask], dim=1)  # item att mask has 1 extra for cls which is removed and sep token which is added.
            seg_seq = self.segment_embedding(
                torch.LongTensor([[0] * (user_input_ids.shape[1] + 1) + [1] * (item_input_ids.shape[1] - 1)]).to(
                    self.device))
            output = self.bert.forward(inputs_embeds=torch.add(concat_ids, seg_seq),
                                       attention_mask=att_mask)
        else:
            sep_tokens = torch.tensor([102], device=self.device).repeat(user_input_ids.shape[0], 1) # TODO sep token
            seg_seq = self.segment_embedding(torch.LongTensor([[0] * (user_input_ids.shape[1]+1) + [1] * (item_input_ids.shape[1]-1)]).to(
                    self.device))
            input_ids = torch.concat([user_input_ids, sep_tokens, item_input_ids[:, 1:]], dim=1)
            att_mask = torch.concat([user_att_mask, item_att_mask], dim=1)  # item att mask has 1 extra for cls which is removed and sep token which is added.
            token_embeddings = torch.add(self.bert_embeddings.forward(input_ids), seg_seq)
            output = self.bert.forward(inputs_embeds=token_embeddings,
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

