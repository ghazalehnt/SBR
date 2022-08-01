import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfile(torch.nn.Module):
    def __init__(self, config, n_users, n_items, num_classes):
        super(VanillaClassifierUserTextProfileItemTextProfile, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(config['pretrained_model'])
        if config['tune_BERT'] is False:
            for param in self.bert.parameters():
                param.requires_grad = False
        bert_embedding_dim = self.bert.embeddings.word_embeddings.weight.shape[1]

        if config["append_id"]:
            self.user_id_embedding = torch.nn.Embedding(n_users, bert_embedding_dim)
            self.item_id_embedding = torch.nn.Embedding(n_items, bert_embedding_dim)
            self.bert_embeddings = self.bert.get_input_embeddings()

        if "k" in config:
            self.transform_u = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.transform_i = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.classifier = torch.nn.Linear(2*config['k'], num_classes)
        else:
            self.classifier = torch.nn.Linear(2 * bert_embedding_dim, num_classes)

        self.agg_strategy = config['agg_strategy']
        self.chunk_agg_strategy = config['chunk_agg_strategy']

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        if hasattr(self, 'bert_embeddings'):  # we want to append an id embedding to the tokens
            batch_size = batch[INTERNAL_USER_ID_FIELD].shape[0]
            # user:
            user_ids = batch[INTERNAL_USER_ID_FIELD]
            id_embeddings = self.user_id_embedding(user_ids)
            outputs = []
            # go over chunks:
            for input_ids, att_mask in zip(
                    batch['user_chunks_input_ids'], batch['user_chunks_attention_mask']):
                token_embeddings = self.bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, id_embeddings], dim=1), other_tokens], dim=1)
                concat_masks = torch.concat([torch.ones((batch_size, 1), device=att_mask.device), att_mask], dim=1)
                output = self.bert.forward(inputs_embeds=concat_ids,
                                           attention_mask=concat_masks)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            user_rep = torch.stack(outputs).max(dim=0).values
            # item:
            item_ids = batch[INTERNAL_ITEM_ID_FIELD]
            id_embeddings = self.item_id_embedding(item_ids)
            outputs = []
            # go over chunks:
            for input_ids, att_mask in zip(
                    batch['item_chunks_input_ids'], batch['item_chunks_attention_mask']):
                token_embeddings = self.bert_embeddings.forward(input_ids)
                cls_tokens = token_embeddings[:, 0].unsqueeze(1)
                other_tokens = token_embeddings[:, 1:]
                # insert user_id embedding after the especial CLS token:
                concat_ids = torch.concat([torch.concat([cls_tokens, id_embeddings], dim=1), other_tokens], dim=1)
                concat_masks = torch.concat([torch.ones((batch_size, 1), device=att_mask.device), att_mask], dim=1)
                output = self.bert.forward(inputs_embeds=concat_ids,
                                           attention_mask=concat_masks)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            item_rep = torch.stack(outputs).max(dim=0).values
        else:
            # user:
            outputs = []
            for input_ids, att_mask in zip(batch['user_chunks_input_ids'], batch['user_chunks_attention_mask']):
                output = self.bert.forward(input_ids=input_ids,
                                           attention_mask=att_mask)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            user_rep = torch.stack(outputs).max(dim=0).values
            # item:
            outputs = []
            for input_ids, att_mask in zip(batch['item_chunks_input_ids'], batch['item_chunks_attention_mask']):
                output = self.bert.forward(input_ids=input_ids,
                                           attention_mask=att_mask)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            item_rep = torch.stack(outputs).max(dim=0).values

        if hasattr(self, 'transform_u'):
            user_rep = self.transform_u(user_rep)
            item_rep = self.transform_i(item_rep)

        result = self.classifier(torch.concat([user_rep, item_rep], dim=1))
        return result  # do not apply sigmoid and use BCEWithLogitsLoss