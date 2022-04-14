import torch
import transformers

from SBR.utils.statics import INTERNAL_USER_ID_FIELD


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

        if "k" in config:
            self.transform_u = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.transform_i = torch.nn.Linear(bert_embedding_dim, config['k'])
            self.classifier = torch.nn.Linear(2*config['k'], num_classes)
        else:
            self.classifier = torch.nn.Linear(2 * bert_embedding_dim, num_classes)

        self.agg_strategy = config['agg_strategy']
        self.chunk_agg_strategy = config['chunk_agg_strategy']

    def forward(self, batch):
        # if hasattr(self, 'user_id_embedding'):
        #     pass
        # else:
        if True:
            # batch -> chunks * batch_size * tokens
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
        return torch.sigmoid(result)