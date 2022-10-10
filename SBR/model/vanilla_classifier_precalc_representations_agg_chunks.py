import os.path

import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(torch.nn.Module):
    def __init__(self, config, n_users, n_items, device, prec_dir, dataset_config):
        super(VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks, self).__init__()
        bert_embedding_dim = 768

        self.transform_u_1 = torch.nn.Linear(bert_embedding_dim, config['k1'])
        self.transform_i_1 = torch.nn.Linear(bert_embedding_dim, config['k1'])
        self.transform_u_2 = torch.nn.Linear(config['k1'], config['k2'])
        self.transform_i_2 = torch.nn.Linear(config['k1'], config['k2'])

        self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
        self.item_bias = torch.nn.Parameter(torch.zeros(n_items))
        self.bias = torch.nn.Parameter(torch.zeros(1))

        self.chunk_agg_strategy = config['chunk_agg_strategy']
        max_num_chunks_user = dataset_config['max_num_chunks_user']
        max_num_chunks_item = dataset_config['max_num_chunks_item']

        if "use_random_reps" in config and config["use_random_reps"] == True:
            self.chunk_user_reps = {}
            for c in range(max_num_chunks_user):
                self.chunk_user_reps[c] = torch.nn.Embedding(n_users, 768, device=device)
                self.chunk_user_reps[c].requires_grad_(False)

            self.chunk_item_reps = {}
            for c in range(max_num_chunks_item):
                self.chunk_item_reps[c] = torch.nn.Embedding(n_items, 768, device=device)
                self.chunk_item_reps[c].requires_grad_(False)
        else:
            user_rep_file = f"user_representation_" \
                            f"{config['agg_strategy']}_" \
                            f"id{config['append_id']}_" \
                            f"tb{config['tune_BERT']}_" \
                            f"cf{config['use_CF']}_" \
                            f"{'-'.join(dataset_config['user_text'])}_" \
                            f"{dataset_config['user_item_text_choice']}_" \
                            f"{dataset_config['user_item_text_tie_breaker'] if dataset_config['user_text_filter'] in ['', 'item_sentence_SBERT'] else ''}_" \
                            f"{dataset_config['user_text_filter'] if len(dataset_config['user_text_filter']) > 0 else 'no-filter'}" \
                            f"{'_i' + '-'.join(dataset_config['item_text']) if dataset_config['user_text_filter'] in ['item_sentence_SBERT'] else ''}" \
                            f".pkl"
            if os.path.exists(os.path.join(prec_dir, user_rep_file)):
                user_chunk_reps = torch.load(os.path.join(prec_dir, user_rep_file), map_location=device)
            else:
                raise ValueError(f"Precalculated user embedding does not exist! {os.path.join(prec_dir, user_rep_file)}")

            self.chunk_user_reps = {}
            for c in range(max_num_chunks_user):
                ch_rep = []
                for user_chunks in user_chunk_reps:
                    if c < len(user_chunks):
                        ch_rep.append(user_chunks[c])
                    else:
                        if self.chunk_agg_strategy == "max_pool":
                            # ch_rep.append(-1 * torch.inf * torch.ones((1, 768)))
                            # ch_rep.append(torch.zeros((1, 768)))
                            ch_rep.append(user_chunks[0])  # todo add the chunk0 or last chunk when no more chunks
                        else:
                            raise NotImplementedError()
                self.chunk_user_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=config['freeze_prec_reps']) # TODO? or concat? stach -> n*1*768 , concatn*768

            item_rep_file = f"item_representation_" \
                            f"{config['agg_strategy']}_" \
                            f"id{config['append_id']}_" \
                            f"tb{config['tune_BERT']}_" \
                            f"cf{config['use_CF']}_" \
                            f"{'-'.join(dataset_config['item_text'])}" \
                            f".pkl"
            if os.path.exists(os.path.join(prec_dir, item_rep_file)):
                item_chunks_reps = torch.load(os.path.join(prec_dir, item_rep_file), map_location=device)
            else:
                raise ValueError(f"Precalculated item embedding does not exist! {os.path.join(prec_dir, item_rep_file)}")

            self.chunk_item_reps = {}
            for c in range(max_num_chunks_item):
                ch_rep = []
                for item_chunks in item_chunks_reps:
                    if c < len(item_chunks):
                        ch_rep.append(item_chunks[c])
                    else:
                        if self.chunk_agg_strategy == "max_pool":
                            ch_rep.append(item_chunks[0])  # todo add the chunk0 or last chunk when no more chunks
                        else:
                            raise NotImplementedError()
                self.chunk_item_reps[c] = torch.nn.Embedding.from_pretrained(torch.concat(ch_rep), freeze=config['freeze_prec_reps'])  # TODO? or concat? stach -> n*1*768 , concatn*768

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        # todo for users/items who had fewer than max-chunks, I put their chunk[0] for the missing chunks, so the output would be same az 0, max pooling gets chunk0
        user_reps = []
        for c in range(len(self.chunk_user_reps.keys())):
            user_ch_rep = self.chunk_user_reps[c](user_ids)
            user_ch_rep = torch.nn.functional.relu(self.transform_u_1(user_ch_rep))
            user_ch_rep = self.transform_u_2(user_ch_rep)
            user_reps.append(user_ch_rep)
        user_reps = torch.stack(user_reps).max(dim=0).values

        item_reps = []
        for c in range(len(self.chunk_item_reps.keys())):
            item_ch_rep = self.chunk_item_reps[c](item_ids)
            item_ch_rep = torch.nn.functional.relu(self.transform_i_1(item_ch_rep))
            item_ch_rep = self.transform_i_2(item_ch_rep)
            item_reps.append(item_ch_rep)
        item_reps = torch.stack(item_reps).max(dim=0).values

        # TODO change later: this is slow
        # user_reps = []
        # for u in user_ids:
        #     outputs = []
        #     for c in range(0, min(len(self.user_chunks_reps[u]), self.max_num_chunks_user)):
        #         ch_rep = self.user_chunks_reps[u][c]
        #         ch_rep = torch.nn.functional.relu(self.transform_u_1(ch_rep))
        #         ch_rep = self.transform_u_2(ch_rep)
        #         outputs.append(ch_rep)
        #     if self.chunk_agg_strategy == "max_pool":
        #         rep = torch.stack(outputs).max(dim=0).values
        #     else:
        #         raise NotImplementedError()
        #     user_reps.append(rep)
        # user_reps = torch.concat(user_reps) ## TODO check
        # item_reps = []
        # for i in item_ids:
        #     outputs = []
        #     for c in range(0, min(len(self.item_chunks_reps[i]), self.max_num_chunks_item)):
        #         ch_rep = self.item_chunks_reps[i][c]
        #         ch_rep = torch.nn.functional.relu(self.transform_i_1(ch_rep))
        #         ch_rep = self.transform_i_2(ch_rep)
        #         outputs.append(ch_rep)
        #     if self.chunk_agg_strategy == "max_pool":
        #         rep = torch.stack(outputs).max(dim=0).values
        #     else:
        #         raise NotImplementedError()
        #     item_reps.append(rep)
        # item_reps = torch.concat(item_reps)  ## TODO check

        result = torch.sum(torch.mul(user_reps, item_reps), dim=1)
        result = result + self.item_bias[item_ids] + self.user_bias[user_ids]
        result = result + self.bias
        result = result.unsqueeze(1)
        return result  # do not apply sigmoid and use BCEWithLogitsLoss

