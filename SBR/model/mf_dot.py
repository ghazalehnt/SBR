import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class MatrixFactorizatoinDotProduct(torch.nn.Module):
    def __init__(self, config, n_users, n_items, device, use_item_bias=False, use_user_bias=False):
        super(MatrixFactorizatoinDotProduct, self).__init__()
        self.item_prec_reps = None
        self.user_prec_reps = None
        self.device = device

        self.user_embedding = torch.nn.Embedding(n_users, config["embedding_dim"])
        self.item_embedding = torch.nn.Embedding(n_items, config["embedding_dim"])
        if "embed_init" in config:
            if config["embed_init"] in ["xavier_uniform", "xavier"]:  # due to initial param values, later "xavier" could be removed 
                torch.nn.init.xavier_uniform_(self.user_embedding.weight)
                torch.nn.init.xavier_uniform_(self.item_embedding.weight)
            elif config["embed_init"] == "xavier_normal":
                torch.nn.init.xavier_normal_(self.user_embedding.weight)
                torch.nn.init.xavier_normal_(self.item_embedding.weight)
            else:
                raise NotImplementedError("embed init not implemented")

        self.use_item_bias = use_item_bias
        self.use_user_bias = use_user_bias

        if self.use_user_bias:
            self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
        if self.use_item_bias:
            self.item_bias = torch.nn.Parameter(torch.zeros(n_items))

    def prec_representations_for_test(self, users, items, padding_token):
        # user:
        dataloader = DataLoader(users, batch_size=1024)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            user_ids = batch[INTERNAL_USER_ID_FIELD].to(self.device)
            user_embeds = self.user_embedding(user_ids)
            reps.extend(user_embeds.tolist())
        self.user_prec_reps = torch.nn.Embedding.from_pretrained(torch.tensor(reps)).to(self.device)

        #item:
        dataloader = DataLoader(items, batch_size=1024)
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        reps = []
        for batch_idx, batch in pbar:
            item_ids = batch[INTERNAL_ITEM_ID_FIELD].to(self.device)
            item_embeds = self.item_embedding(item_ids)
            reps.extend(item_embeds.tolist())
        self.item_prec_reps = torch.nn.Embedding.from_pretrained(torch.tensor(reps)).to(self.device)


    def forward(self, batch):
        users = batch[INTERNAL_USER_ID_FIELD].squeeze()
        items = batch[INTERNAL_ITEM_ID_FIELD].squeeze()

        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding(items)

        # we want to multiply each user with its corresponding item:  # todo which is faster? time these? very similar not much different
        # 1: elementwise multiplication of user and item embeds and them sum on dim1
        output = torch.sum(torch.mul(user_embeds, item_embeds), dim=1)
        # 2: taking the diagonal of matrixmul of user and item embeds:
        #output = torch.diag(torch.matmul(user_embeds, item_embeds.T))
        if self.use_item_bias:
            output = output + self.item_bias[items]
        if self.use_user_bias:
            output = output + self.user_bias[users]
        return output.unsqueeze(1)  # do not apply sigmoid and use BCEWithLogitsLoss
