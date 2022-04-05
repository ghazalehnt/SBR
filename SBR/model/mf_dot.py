import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class MatrixFactorizatoinDotProduct(torch.nn.Module):
    def __init__(self, config, n_users, n_items):
        super(MatrixFactorizatoinDotProduct, self).__init__()

        self.user_embedding = torch.nn.Embedding(n_users, config["embedding_dim"])
        self.item_embedding = torch.nn.Embedding(n_items, config["embedding_dim"])
        self.user_bias = torch.nn.Parameter(torch.zeros(n_users))
        self.item_bias = torch.nn.Parameter(torch.zeros(n_items))
        self.bias = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch):
        users = batch[INTERNAL_USER_ID_FIELD]
        items = batch[INTERNAL_ITEM_ID_FIELD]

        user_embeds = self.user_embedding(users)
        item_embeds = self.item_embedding(items)

        # we want to multiply each user with its corresponding item:  # todo which is faster? time these? very similar not much different
        # 1: elementwise multiplication of user and item embeds and them sum on dim1
        output = torch.sum(torch.mul(user_embeds, item_embeds), dim=1)
        # 2: taking the diagonal of matrixmul of user and item embeds:
        #output = torch.diag(torch.matmul(user_embeds, item_embeds.T))
        output = output + self.item_bias[items] + self.user_bias[users]
        output = output + self.bias
        return torch.sigmoid(output)  # apply sigmoid and use BCELoss
        # return output   # OR do not apply sigmoid, .. as BCELossWithLogits does that already
