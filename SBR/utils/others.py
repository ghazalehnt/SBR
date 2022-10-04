from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.vanilla_classifier import VanillaClassifierUserTextProfileItemTextProfile
from SBR.model.vanilla_classifier_precalc_representations import VanillaClassifierUserTextProfileItemTextProfilePrecalculated
from SBR.model.mf_dot_text import MatrixFactorizatoinTextDotProduct
from vanilla_classifier_precalc_representations_agg_chunks import \
    VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks


def get_model(config, user_info, item_info, n_classes=None, padding_token=None, device=None, prec_dir=None,
              dataset_config=None):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0])
    elif config['name'] == "VanillaClassifier":
        model = VanillaClassifierUserTextProfileItemTextProfile(config=config, n_users=user_info.shape[0],
                                                                n_items=item_info.shape[0], num_classes=n_classes)
    elif config['name'] == "VanillaClassifier_precalc":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculated(config=config,
                                                                             n_users=user_info.shape[0],
                                                                             n_items=item_info.shape[0],
                                                                             num_classes=n_classes,
                                                                             user_info=user_info, item_info=item_info,
                                                                             padding_token=padding_token,
                                                                             device=device,
                                                                             prec_dir=prec_dir,
                                                                             dataset_config=dataset_config)
    elif config['name'] == "MF_TEXT_DOT":
        model = MatrixFactorizatoinTextDotProduct(config=config,
                                                  n_users=user_info.shape[0],
                                                  n_items=item_info.shape[0],
                                                  device=device,
                                                  prec_dir=prec_dir,
                                                  dataset_config=dataset_config)
    elif config['name'] == "VanillaClassifier_precalc_agg_chunks":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(config=config,
                                                                                      n_users=user_info.shape[0],
                                                                                      n_items=item_info.shape[0],
                                                                                      device=device,
                                                                                      prec_dir=prec_dir,
                                                                                      dataset_config=dataset_config)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
