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
        "up_sr_sf": "userprofile.interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.category-item.description",
        "up_sr_sb": "userprofile.interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.category",
        "up_sr_i": "userprofile.interaction.summary-interaction.reviewText_idf_sentence",
        "up_sr_3": "userprofile.interaction.summary-interaction.reviewText_tf-idf_3",
        "up_t_r": "userprofile.item.title_random_sentence",
        "up_t_sf": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.category-item.description",
        "up_t_sb": "userprofile.item.title_item_sentence_SBERT_iitem.title-item.category",
        "up_t_i": "userprofile.item.title_idf_sentence",
        "up_t_3": "userprofile.item.title_tf-idf_3",
        "up_tsr_r": "userprofile.item.title-interaction.summary-interaction.reviewText_random_sentence",
        "up_tsr_sf": "userprofile.item.title-interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.category-item.description",
        "up_tsr_sb": "userprofile.item.title-interaction.summary-interaction.reviewText_item_sentence_SBERT_iitem.title-item.category",
        "up_tsr_i": "userprofile.item.title-interaction.summary-interaction.reviewText_idf_sentence",
        "up_tsr_3": "userprofile.item.title-interaction.summary-interaction.reviewText_tf-idf_3",
    },
    "GR_UCSD": {
        "t": "item.title",
        "g": "item.genres",
        "d": "item.description",
        "r": "interaction.review_text",
        "m": "interaction.model_keywords",
        "i": "interaction.instructional_keywords",
        "t5": "interaction.t5",
        "ch": "user.chatgpt",
        "s": "user.summarizer",
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
        "up_ik_r": "userprofile.interaction.instructional_keywords_random_sentence",
        "up_mk_r": "userprofile.interaction.model_keywords_random_sentence",
        "up_ch_": "userprofile.user.chatgpt_",
        "up_sum_": "userprofile.user.summarizer_",
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
