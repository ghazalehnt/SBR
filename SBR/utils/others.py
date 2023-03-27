from SBR.model.DeepCoNN import DeepCoNN
from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.vanilla_classifier_precalc_representations_agg_chunks import \
    VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks
from vanilla_classifier_precalc_representations_agg_chunks_end_to_end import \
    VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunksEndToEnd


def get_model(config, user_info, item_info, device=None, dataset_config=None, exp_dir=None):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=False, use_user_bias=False)
    elif config['name'] == "MF_with_itembias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=False)
    elif config['name'] == "MF_with_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=False, use_user_bias=True)
    elif config['name'] == "MF_with_itembias_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=True)
    elif config['name'] == "VanillaBERT_precalc_embed_sim":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=False,
                                                                                      use_item_bias=False,
                                                                                      use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_ffn":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=True,
                                                                                      use_item_bias=False,
                                                                                      use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_transformersinglenew":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=False,
                                                                                      use_item_bias=False,
                                                                                      use_user_bias=False,
                                                                                      use_transformer=True)
    elif config['name'] == "VanillaBERT_precalc_with_transformersinglenew_endtoend":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunksEndToEnd(model_config=config,
                                                                                              users=user_info,
                                                                                              items=item_info,
                                                                                              device=device,
                                                                                              dataset_config=dataset_config,
                                                                                              use_ffn=False,
                                                                                              use_item_bias=False,
                                                                                              use_user_bias=False,
                                                                                              use_transformer=True)
    elif config['name'] == "VanillaBERT_precalc_with_ffn_endtoend":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunksEndToEnd(model_config=config,
                                                                                              users=user_info,
                                                                                              items=item_info,
                                                                                              device=device,
                                                                                              dataset_config=dataset_config,
                                                                                              use_ffn=True,
                                                                                              use_item_bias=False,
                                                                                              use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_itembias":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=False,
                                                                                      use_item_bias=True,
                                                                                      use_user_bias=False)
    elif config['name'] == "VanillaBERT_precalc_with_ffn_itembias":
        model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
                                                                                      users=user_info,
                                                                                      items=item_info,
                                                                                      device=device,
                                                                                      dataset_config=dataset_config,
                                                                                      use_ffn=True,
                                                                                      use_item_bias=True,
                                                                                      use_user_bias=False)
    elif config['name'] == "DeepCoNN":
        model = DeepCoNN(config, exp_dir)
    else:
        raise ValueError(f"Model is not implemented! model.name = {config['name']}")
    return model
