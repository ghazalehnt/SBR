from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.vanilla_classifier import VanillaClassifierUserTextProfileItemTextProfile
from SBR.model.vanilla_classifier_precalc_representations import VanillaClassifierUserTextProfileItemTextProfilePrecalculated


def get_model(config, user_info, item_info, n_classes=None):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0])
    elif config['name'] == "VanillaClassifier":
        model = VanillaClassifierUserTextProfileItemTextProfile(config=config, n_users=user_info.shape[0],
                                                                n_items=item_info.shape[0], num_classes=n_classes)
    elif config['name'] == "VanillaClassifier_precalc":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculated(config=config, n_users=user_info.shape[0],
                                                                             n_items=item_info.shape[0],
                                                                             num_classes=n_classes,
                                                                             user_info=user_info, item_info=item_info)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
