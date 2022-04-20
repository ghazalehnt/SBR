import time

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD
from SBR.utils.data_loading import CollateRepresentationBuilder


class VanillaClassifierUserTextProfileItemTextProfilePrecalculated(torch.nn.Module):
    def __init__(self, config, n_users, n_items, num_classes, user_info, item_info):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculated, self).__init__()
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
        self.batch_size = config['precalc_batch_size']

        start = time.time()
        self.user_rep = self.create_representations(user_info)
        print(f"user rep loaded in {time.time()-start}")
        start = time.time()
        self.item_rep = self.create_representations(item_info)
        print(f"item rep loaded in {time.time()-start}")

    def create_representations(self, info):
        # TODO tokenizer padding pass to collate fn
        collate_fn = CollateRepresentationBuilder()
        dataloader = DataLoader(info, batch_size=self.batch_size, collate_fn=collate_fn)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            # go over chunks:
            outputs = []
            for input_ids, att_mask in zip(batch['chunks_input_ids'], batch['chunks_attention_mask']):
                output = self.bert.forward(input_ids=input_ids,
                                           attention_mask=att_mask)
                if self.agg_strategy == "CLS":
                    temp = output.pooler_output
                elif self.agg_strategy == "mean":
                    raise ValueError("not implemented yet")
                else:
                    raise ValueError(f"agg_strategy not implemented {self.agg_strategy}")
                outputs.append(temp)
            rep = torch.stack(outputs).max(dim=0).values
            reps.append(rep)
        return torch.nn.Embedding.from_pretrained(torch.concat(reps), freeze=True)  # todo freeze? or unfreeze?

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze()
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze()
        user_rep = self.user_rep(user_ids)
        item_rep = self.item_rep(item_ids)
        if hasattr(self, 'transform_u'):
            user_rep = self.transform_u(user_rep)
            item_rep = self.transform_i(item_rep)
        result = self.classifier(torch.concat([user_rep, item_rep], dim=1))
        return result  # do not apply sigmoid and use BCEWithLogitsLoss

