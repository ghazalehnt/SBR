from SBR.model.DeepCoNN import DeepCoNN
from SBR.model.bert_ffn_prec_rep_agg_chunk import BertFFNPrecomputedRepsChunkAgg
from SBR.model.mf_dot import MatrixFactorizatoinDotProduct
from SBR.model.vanilla_classifier_precalc_representations_agg_chunks import \
    VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks
from SBR.model.vanilla_classifier_end_to_end import \
    VanillaClassifierUserTextProfileItemTextProfileEndToEnd
from SBR.model.vanilla_classifier_joint_end_to_end import \
    VanillaClassifierUserTextProfileItemTextProfileJointEndToEnd
from SBR.model.bert_ffn_end_to_end import BertFFNUserTextProfileItemTextProfileEndToEnd
from SBR.model.bert_singleffn_end_to_end import BertSignleFFNUserTextProfileItemTextProfileEndToEnd
from SBR.model.cf_ffn_dot import CFFFNDOT
from SBR.model.bert_singleffn_prec_agg_chunk import BertSignleFFNPrecomputedRepsChunkAgg


def get_model(config, user_info, item_info, device=None, dataset_config=None, exp_dir=None, test_only=False):
    if config['name'] == "MF":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              device=device, use_item_bias=False, use_user_bias=False)
    elif config['name'] == "MF_ffn":
        model = CFFFNDOT(model_config=config, n_users=user_info.shape[0], n_items=item_info.shape[0], device=device)
    elif config['name'] == "MF_with_itembias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=False)
    elif config['name'] == "MF_with_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=False, use_user_bias=True)
    elif config['name'] == "MF_with_itembias_userbias":
        model = MatrixFactorizatoinDotProduct(config=config, n_users=user_info.shape[0], n_items=item_info.shape[0],
                                              use_item_bias=True, use_user_bias=True)
    # elif config['name'] == "VanillaBERT_precalc_embed_sim":
    #     model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
    #                                                                                   users=user_info,
    #                                                                                   items=item_info,
    #                                                                                   device=device,
    #                                                                                   dataset_config=dataset_config)
    # elif config['name'] == "VanillaBERT_precalc_with_ffn":
    #     model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
    #                                                                                   users=user_info,
    #                                                                                   items=item_info,
    #                                                                                   device=device,
    #                                                                                   dataset_config=dataset_config,
    #                                                                                   use_ffn=True)
    # elif config['name'] == "VanillaBERT_precalc_with_transformersinglenew":
    #     model = VanillaClassifierUserTextProfileItemTextProfilePrecalculatedAggChunks(model_config=config,
    #                                                                                   users=user_info,
    #                                                                                   items=item_info,
    #                                                                                   device=device,
    #                                                                                   dataset_config=dataset_config,
    #                                                                                   use_transformer=True)
    elif config['name'] == "VanillaBERT_precalc_ffn":
        model = BertFFNPrecomputedRepsChunkAgg(model_config=config,
                                               device=device,
                                               dataset_config=dataset_config,
                                               users=user_info,
                                               items=item_info)
    elif config['name'] == "VanillaBERT_precalc_1ffn":
        model = BertSignleFFNPrecomputedRepsChunkAgg(model_config=config,
                                                     device=device,
                                                     dataset_config=dataset_config,
                                                     users=user_info,
                                                     items=item_info)
    elif config['name'] == "VanillaBERT_transformersinglenew_endtoend":
        model = VanillaClassifierUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                                        users=user_info,
                                                                        items=item_info,
                                                                        device=device,
                                                                        dataset_config=dataset_config,
                                                                        use_transformer=True)
    elif config['name'] == "VanillaBERT_ffn_endtoend":
        model = BertFFNUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                              device=device,
                                                              dataset_config=dataset_config,
                                                              users=user_info,
                                                              items=item_info,
                                                              test_only=test_only)
    elif config['name'] == "VanillaBERT_1ffn_endtoend":
        model = BertSignleFFNUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                                    device=device,
                                                                    dataset_config=dataset_config)
    elif config['name'] == "VanillaBERT_endtoend_dotproduct":
        model = VanillaClassifierUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                                        users=user_info,
                                                                        items=item_info,
                                                                        device=device,
                                                                        dataset_config=dataset_config)
    elif config['name'] == "VanillaBERT_endtoend_mlp":
        model = VanillaClassifierUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                                        users=user_info,
                                                                        items=item_info,
                                                                        device=device,
                                                                        dataset_config=dataset_config,
                                                                        use_mlp=True)
    elif config['name'] == "VanillaBERT_endtoend_fm":
        model = VanillaClassifierUserTextProfileItemTextProfileEndToEnd(model_config=config,
                                                                        users=user_info,
                                                                        items=item_info,
                                                                        device=device,
                                                                        dataset_config=dataset_config,
                                                                        use_factorization=True)
    elif config['name'] == "VanillaBERT_endtoend_joint":
        model = VanillaClassifierUserTextProfileItemTextProfileJointEndToEnd(model_config=config,
                                                                             users=user_info,
                                                                             items=item_info,
                                                                             device=device,
                                                                             dataset_config=dataset_config)
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
