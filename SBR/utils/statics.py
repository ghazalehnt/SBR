INTERNAL_USER_ID_FIELD = "internal_user_id"
INTERNAL_ITEM_ID_FIELD = "internal_item_id"

user_item_text = {
    "Amazon": {
        "t": "item.title",
        "c": "item.category",
        "d": "item.description",
        "s": "interaction.summary",
        "r": "interaction.reviewText",
        "g10": "user.sorted_genres_10",
        "ug": "user.sorted_genres",
        "up_sr_r": "userprofile.interaction.summary-interaction.reviewText_random_sentence",
        "up_sr_sf": "userprofile.interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_sr_sb": "userprofile.interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.genres",
        "up_sr_i": "userprofile.interaction.summary-interaction.reviewText_idf_sentence",
        "up_sr_3": "userprofile.interaction.summary-interaction.reviewText_tf-idf_3",
        "up_t_r": "userprofile.item.title_random_sentence",
        "up_t_sf": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_t_sb": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.genres",
        "up_t_i": "userprofile.item.title_idf_sentence",
        "up_t_3": "userprofile.item.title_tf-idf_3",
        "up_tsr_r": "userprofile.item.title-interaction.summary-interaction.reviewText_random_sentence",
        "up_tsr_sf": "userprofile.item.title-interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_tsr_sb": "userprofile.item.title-interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.genres",
        "up_tsr_i": "userprofile.item.title-interaction.summary-interaction.reviewText_idf_sentence",
        "up_tsr_3": "userprofile.item.title-interaction.summary-interaction.reviewText_tf-idf_3",

        "mk_sr": "interaction.model_keywords_sr",
        "ch_sr": "user.chatgpt_sr",
        "s_sr": "user.summarizer_sr",
        "mk_t": "interaction.model_keywords_t",
        "ch_t": "user.chatgpt_t",
        "s_t": "user.summarizer_t",
        "mk_tsr": "interaction.model_keywords_tsr",
        "ch_tsr": "user.chatgpt_tsr",
        "s_tsr": "user.summarizer_tsr",

        "up_mk_sr_r": "userprofile.interaction.model_keywords_sr_random_sentence",
        "up_ch_sr_": "userprofile.user.chatgpt_sr_",
        "up_sum_sr_": "userprofile.user.summarizer_sr_",
        "up_mk_t_r": "userprofile.interaction.model_keywords_t_random_sentence",
        "up_ch_t_": "userprofile.user.chatgpt_t_",
        "up_sum_t_": "userprofile.user.summarizer_t_",
        "up_mk_tsr_r": "userprofile.interaction.model_keywords_tsr_random_sentence",
        "up_ch_tsr_": "userprofile.user.chatgpt_tsr_",
        "up_sum_tsr_": "userprofile.user.summarizer_tsr_",
    },
    "GR_UCSD": {
        "t": "item.title",
        "g": "item.genres",
        "d": "item.description",
        "r": "interaction.review_text",
        "g5": "user.sorted_genres_5",
        "ug": "user.sorted_genres",
        "up_r_r": "userprofile.interaction.review_text_random_sentence",
        "up_r_sf": "userprofile.interaction.review_text_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_r_sb": "userprofile.interaction.review_text_item_sentence_SBERT_iitem.title-item.genres",
        "up_r_i": "userprofile.interaction.review_text_idf_sentence",
        "up_r_3": "userprofile.interaction.review_text_tf-idf_3",
        "up_t_r": "userprofile.item.title_random_sentence",
        "up_t_sf": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_t_sb": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.genres",
        "up_t_i": "userprofile.item.title_idf_sentence",
        "up_t_3": "userprofile.item.title_tf-idf_3",
        "up_tr_r": "userprofile.item.title-interaction.review_text_random_sentence",
        "up_tr_sf": "userprofile.item.title-interaction.review_text_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_tr_sb": "userprofile.item.title-interaction.review_text_item_sentence_SBERT_iitem.title-item.genres",
        "up_tr_i": "userprofile.item.title-interaction.review_text_idf_sentence",
        "up_tr_3": "userprofile.item.title-interaction.review_text_tf-idf_3",
        "up_d_r": "userprofile.item.description_random_sentence",
        "up_d_sf": "userprofile.item.description_item_sentence_SBERT_iitem.title-item.genres-item.description",
        "up_d_i": "userprofile.item.description_idf_sentence",
        "up_d_3": "userprofile.item.description_tf-idf_3",

        "mk_r": "interaction.model_keywords_r",
        "ch_r": "user.chatgpt_r",
        "s_r": "user.summarizer_r",
        "mk_t": "interaction.model_keywords_t",
        "ch_t": "user.chatgpt_t",
        "s_t": "user.summarizer_t",
        "mk_tr": "interaction.model_keywords_tr",
        "ch_tr": "user.chatgpt_tr",
        "s_tr": "user.summarizer_tr",

        "up_mk_r_r": "userprofile.interaction.model_keywords_r_random_sentence",
        "up_ch_r_": "userprofile.user.chatgpt_r_",
        "up_sum_r_": "userprofile.user.summarizer_r_",
        "up_mk_t_r": "userprofile.interaction.model_keywords_t_random_sentence",
        "up_ch_t_": "userprofile.user.chatgpt_t_",
        "up_sum_t_": "userprofile.user.summarizer_t_",
        "up_mk_tr_r": "userprofile.interaction.model_keywords_tr_random_sentence",
        "up_ch_tr_": "userprofile.user.chatgpt_tr_",
        "up_sum_tr_": "userprofile.user.summarizer_tr_",
    },
    "CGR": {
        "t": "item.title",
        "g": "item.genres",
        "d": "item.description",
        "r": "interaction.review"
    },
}


def get_rev_map(dataset):
    return {v:k for k, v in user_item_text[dataset].items()}


def get_profile(dataset, shortened):
    global user_item_text
    return [user_item_text[dataset][i] for i in shortened.split("-")]
