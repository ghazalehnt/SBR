from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model. vanilla_classifier import VanillaClassifier


def get_model(config, n_users, n_items, n_classes=None):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=n_users, n_items=n_items)
    elif config['name'] == "VanillaClassifier":
        model = VanillaClassifier(config=config, n_users=n_users, n_items=n_items, num_classes=n_classes)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
