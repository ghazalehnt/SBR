import torch
import transformers


class VanillaClassifier(torch.nn.Module):
    def __init__(self, config, n_users, n_items, num_classes):
        super(VanillaClassifier, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(config['pretrained_model'])
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
        self.max_user_chunks = config['max_num_chunks_user']
        self.max_item_chunks = config['max_num_chunks_item']

    def forward(self, batch):
        # if hasattr(self, 'user_id_embedding'):
        #     pass
        # else:
        if True:
            batch_size = batch['user_input_ids'].shape[0]
            device = batch['user_input_ids'].device
            cls = torch.ones((batch_size, 1), dtype=torch.int64, device=device) * CLS_token
            ones = torch.ones((batch_size, 1), dtype=torch.int64, device=device)
            # user:
            ch_size = 512 - 1  # adding CLS token, however there is one CLS token there
            total_len = batch['user_input_ids'].shape[1]
            num_chunks = min(self.max_user_chunks, (total_len - 1 // ch_size) + 1) #todo talk to andrew, how big is ok?
            CLS_token = batch['user_input_ids'][0][0].item()
            outputs = []
            for ch in range(num_chunks):
                start = 1 + (ch * ch_size)   # +1-> bcs of the initial CLS token
                end = start + ch_size
                input_ids = torch.concat([cls, batch['user_input_ids'][:, start:end]], dim=1)
                att_mast = torch.concat([ones, batch['user_attention_mask'][:, start:end]], dim=1)
                output = self.bert.forward(input_ids=input_ids,
                                           attention_mask=att_mast)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            user_rep = torch.stack(outputs).max(dim=0).values
            # item:
            ch_size = 512 - 1  # adding CLS token, however there is one CLS token there
            total_len = batch['item_input_ids'].shape[1]
            num_chunks = min(self.max_user_chunks, (total_len - 1 // ch_size) + 1) #todo talk to andrew, how big is ok?
            CLS_token = batch['item_input_ids'][0][0].item()
            outputs = []
            for ch in range(num_chunks):
                start = 1 + (ch * ch_size)  # +1-> bcs of the initial CLS token
                end = start + ch_size
                input_ids = torch.concat([cls, batch['item_input_ids'][:, start:end]], dim=1)
                att_mast = torch.concat([ones, batch['item_attention_mask'][:, start:end]], dim=1)
                output = self.bert.forward(input_ids=input_ids,
                                           attention_mask=att_mast)
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