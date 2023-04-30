import torch

from SBR.utils.statics import INTERNAL_USER_ID_FIELD, INTERNAL_ITEM_ID_FIELD


class CFFFNDOT(torch.nn.Module):
    def __init__(self, model_config, n_users, n_items, device):
        super(CFFFNDOT, self).__init__()
        self.device = device

        self.user_embedding = torch.nn.Embedding(n_users, model_config["user_embedding"])
        self.item_embedding = torch.nn.Embedding(n_items, model_config["item_embedding"])

        self.user_linear_layers = [torch.nn.Linear(model_config["user_embedding"], model_config["user_k"][0], device=self.device)]
        for k in range(1, len(model_config["user_k"])):
            self.user_linear_layers.append(torch.nn.Linear(model_config["user_k"][k-1], model_config["user_k"][k], device=self.device))

        self.item_linear_layers = [torch.nn.Linear(model_config["item_embedding"], model_config["item_k"][0], device=self.device)]
        for k in range(1, len(model_config["item_k"])):
            self.item_linear_layers.append(torch.nn.Linear(model_config["item_k"][k - 1], model_config["item_k"][k], device=self.device))

    def forward(self, batch):
        users = batch[INTERNAL_USER_ID_FIELD].squeeze()
        items = batch[INTERNAL_ITEM_ID_FIELD].squeeze()

        user_rep = self.user_embedding(users)
        for k in range(len(self.user_linear_layers) - 1):
            user_rep = torch.nn.functional.relu(self.user_linear_layers[k](user_rep))
        user_rep = self.user_linear_layers[-1](user_rep)

        item_rep = self.item_embedding(items)
        for k in range(len(self.item_linear_layers) - 1):
            item_rep = torch.nn.functional.relu(self.item_linear_layers[k](item_rep))
        item_rep = self.item_linear_layers[-1](item_rep)

        output = torch.sum(torch.mul(user_rep, item_rep), dim=1)
        return output.unsqueeze(1)  # do not apply sigmoid and use BCEWithLogitsLoss
