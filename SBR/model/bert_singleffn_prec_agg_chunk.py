import torch
import os
from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class BertSignleFFNPrecomputedRepsChunkAgg(torch.nn.Module):
    def __init__(self, model_config, device, dataset_config, users, items):
        super(BertSignleFFNPrecomputedRepsChunkAgg, self).__init__()
        if "tune_BERT" in model_config and model_config["tune_BERT"] is True:
            raise ValueError("tune_BERT must be False")
        if "use_CF" in model_config and model_config["use_CF"]:
            raise ValueError("use_CF cannot be true")
        if "append_embedding_to_text" in model_config and model_config["append_embedding_to_text"]:
            raise ValueError("append_embedding_to_text cannot be true")

        bert_embedding_dim = 768
        self.device = device

        chunk_agg_strategy = model_config['chunk_agg_strategy']
        max_user_chunks = dataset_config['max_num_chunks_user'] if "max_num_chunks_user" in dataset_config else None
        if max_user_chunks in ["all", ""]:
            max_user_chunks = None
        max_item_chunks = dataset_config['max_num_chunks_item'] if "max_num_chunks_item" in dataset_config else None
        if max_item_chunks in ["all", ""]:
            max_item_chunks = None

        dim1 = 2 * bert_embedding_dim
        linear_layers = [torch.nn.Linear(dim1, model_config["k"][0], device=self.device)]
        for k in range(1, len(model_config["k"])):
            linear_layers.append(torch.nn.Linear(model_config["k"][k-1], model_config["k"][k], device=self.device))
        linear_layers.append(torch.nn.Linear(model_config['k'][-1], 1, device=self.device))
        self.ffn = torch.nn.ModuleList(linear_layers)

        cf_emb_dim = ""
        if model_config['use_CF']:
            cf_emb_dim = f"_MF-{model_config['CF_embedding_dim']}"

        user_prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
                                      f"size{dataset_config['user_chunk_size']}_"
                                      f"cs-{dataset_config['case_sensitive']}_"
                                      f"nn-{dataset_config['normalize_negation']}_"
                                      f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
        user_rep_file = f"user_representation_" \
                        f"{model_config['agg_strategy']}_" \
                        f"id{model_config['append_id']}_" \
                        f"cf{model_config['use_CF']}_" \
                        f"{dataset_config['user_text_file_name']}" \
                        f".pkl"
        if os.path.exists(os.path.join(user_prec_path, user_rep_file)):
            user_chunk_reps_dict = torch.load(os.path.join(user_prec_path, user_rep_file),
                                              map_location=torch.device('cpu'))
            user_reps = []
            cnt = 0
            for u in users.sort('user_id'):
                ex_user_id = u["user_id"]
                if cnt != u[INTERNAL_USER_ID_FIELD]:
                    raise RuntimeError("wrong user!")
                user_chunks = user_chunk_reps_dict[ex_user_id]
                if max_user_chunks is not None:
                    user_chunks = user_chunk_reps_dict[ex_user_id][:max_user_chunks]

                if chunk_agg_strategy == "max_pool":
                    user_reps.append(torch.stack(user_chunks).max(dim=0).values)
                elif chunk_agg_strategy == "avg":
                    user_reps.append(torch.stack(user_chunks).mean(dim=0))
                cnt += 1
            self.user_reps = torch.nn.Embedding.from_pretrained(torch.concat(user_reps),freeze=True)
        else:
            raise ValueError(
                f"Precalculated user embedding does not exist! {os.path.join(user_prec_path, user_rep_file)}")

        item_prec_path = os.path.join(dataset_config['dataset_path'], f'precomputed_reps{cf_emb_dim}',
                                      f"size{dataset_config['item_chunk_size']}_"
                                      f"cs-{dataset_config['case_sensitive']}_"
                                      f"nn-{dataset_config['normalize_negation']}_"
                                      f"{dataset_config['limit_training_data'] if len(dataset_config['limit_training_data']) > 0 else 'no-limit'}")
        item_rep_file = f"item_representation_" \
                        f"{model_config['agg_strategy']}_" \
                        f"id{model_config['append_id']}_" \
                        f"cf{model_config['use_CF']}_" \
                        f"{dataset_config['item_text_file_name']}" \
                        f".pkl"
        if os.path.exists(os.path.join(item_prec_path, item_rep_file)):
            item_chunks_reps_dict = torch.load(os.path.join(item_prec_path, item_rep_file),
                                               map_location=torch.device('cpu'))
            item_reps = []
            cnt = 0
            for i in items.sort('item_id'):
                ex_item_id = i["item_id"]
                if cnt != i[INTERNAL_ITEM_ID_FIELD]:
                    raise RuntimeError("wrong item!")
                item_chunks = item_chunks_reps_dict[ex_item_id]
                if max_item_chunks is not None:
                    item_chunks = item_chunks_reps_dict[ex_item_id][:max_item_chunks]

                if chunk_agg_strategy == "max_pool":
                    item_reps.append(torch.stack(item_chunks).max(dim=0).values)
                elif chunk_agg_strategy == "avg":
                    item_reps.append(torch.stack(item_chunks).mean(dim=0))
                cnt += 1
            self.item_reps = torch.nn.Embedding.from_pretrained(torch.concat(item_reps), freeze=True)
        else:
            raise ValueError(
                f"Precalculated item embedding does not exist! {os.path.join(item_prec_path, item_rep_file)}")

    def forward(self, batch):
        # batch -> chunks * batch_size * tokens
        user_ids = batch[INTERNAL_USER_ID_FIELD].squeeze(1)
        item_ids = batch[INTERNAL_ITEM_ID_FIELD].squeeze(1)

        user_rep = self.user_reps(user_ids)
        item_rep = self.item_reps(item_ids)

        result = torch.cat((user_rep, item_rep), dim=1)
        for k in range(len(self.ffn) - 1):
            result = torch.nn.functional.relu(self.ffn[k](result))
        result = self.ffn[-1](result)
        return result  # do not apply sigmoid here, later in the trainer if we wanted we would

