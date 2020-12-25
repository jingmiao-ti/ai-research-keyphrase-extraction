# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

# 实现论文第三部分，对候选短语进行排序，并利用MMR调整提取的关键短语的多样性

import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from swisscom_ai.research_keyphrase.model.methods_embeddings import extract_candidates_embedding_for_doc, \
    extract_doc_embedding, extract_sent_candidates_embedding_for_doc


def _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold):
    
    """
    论文第四部分EmbedRank++: Increasing Keyphrase Diversity with MMR
    Core method using Maximal Marginal Relevance in charge to return the top-N candidates

    :param embdistrib: embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param candidates: list of candidates (string)  候选词组 
    :param X: numpy array with the embedding of each candidate in each row 候选词组的embedding矩阵
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of candidates to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """

    N = min(N, len(candidates))
    doc_embedd = extract_doc_embedding(embdistrib, text_obj, use_filtered)  # Extract doc embedding
    doc_sim = cosine_similarity(X, doc_embedd.reshape(1, -1)) # 计算候选单词与文档的相似性 number of candidate phrases,1 

    doc_sim_norm = doc_sim/np.max(doc_sim)  # 公式（3b）
    doc_sim_norm = 0.5 + (doc_sim_norm - np.average(doc_sim_norm)) / np.std(doc_sim_norm)  # 公式（3a） 相似度标准化

    sim_between = cosine_similarity(X)  # 候选单词之间的相似性 number of candidate phrases，number of candidate phrases
    np.fill_diagonal(sim_between, np.NaN)  # 将候选单词的相似性矩阵的对角线设置为 NaN

    sim_between_norm = sim_between/np.nanmax(sim_between, axis=0)
    sim_between_norm = \
        0.5 + (sim_between_norm - np.nanmean(sim_between_norm, axis=0)) / np.nanstd(sim_between_norm, axis=0)

    selected_candidates = []
    unselected_candidates = [c for c in range(len(candidates))]
    
    
    # 挑选第一个关键词，与文档相似性最大的
    j = np.argmax(doc_sim)
    selected_candidates.append(j)
    unselected_candidates.remove(j)
    
    # 依次选择其余关键词
    for _ in range(N - 1):
        selec_array = np.array(selected_candidates)  # 已选关键词与文档相似性最大的索引
        unselec_array = np.array(unselected_candidates)  

        distance_to_doc = doc_sim_norm[unselec_array, :]  # 其余候选关键词与文档的相似性
        dist_between = sim_between_norm[unselec_array][:, selec_array]  # 其余候选词关键词与已选关键词的相似性
        if dist_between.ndim == 1:  # 若其余候选关键词只剩下一个，dist_between的维度为一 []
            dist_between = dist_between[:, np.newaxis]  # 插入新维度[[]]
        j = np.argmax(beta * distance_to_doc - (1 - beta) * np.max(dist_between, axis=1).reshape(-1, 1))  # 选择下一个关键词  公式(2) 考虑了多样性和相关性
        item_idx = unselected_candidates[j]
        selected_candidates.append(item_idx)
        unselected_candidates.remove(item_idx)

    # Not using normalized version of doc_sim for computing relevance
    relevance_list = max_normalization(doc_sim[selected_candidates]).tolist()  # each value is multiplied by 1/max value
    aliases_list = get_aliases(sim_between[selected_candidates, :], candidates, alias_threshold)  # Find candidates which are very similar to the keyphrases (aliases) 返回已选关键词与候选词之间的相似度大于阈值的候选关键词列表，即比较相似的

    return candidates[selected_candidates].tolist(), relevance_list, aliases_list


def MMRPhrase(embdistrib, text_obj, beta=0.65, N=10, use_filtered=True, alias_threshold=0.8):
    """
    Extract N keyphrases

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of keyphrases to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: A tuple with 3 elements :
    1)list of the top-N candidates (or less if there are not enough candidates) (list of string)
    2)list of associated relevance scores (list of float)
    3)list containing for each keyphrase a list of alias (list of list of string)
    """
    candidates, X = extract_candidates_embedding_for_doc(embdistrib, text_obj)  # 候选词组以及嵌入矩阵

    if len(candidates) == 0:
        warnings.warn('No keyphrase extracted for this document')
        return None, None, None

    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered, alias_threshold)


def MMRSent(embdistrib, text_obj, beta=0.5, N=10, use_filtered=True):
    """

    Extract N key sentences

    :param embdistrib: embedding distributor see @EmbeddingDistributor
    :param text_obj: Input text representation see @InputTextObj
    :param beta: hyperparameter beta for MMR (control tradeoff between informativeness and diversity)
    :param N: number of key sentences to extract
    :param use_filtered: if true filter the text by keeping only candidate word before computing the doc embedding
    :return: list of N key sentences (or less if there are not enough candidates)
    """
    candidates, X = extract_sent_candidates_embedding_for_doc(embdistrib, text_obj)

    if len(candidates) == 0:
        warnings.warn('No keysentence extracted for this document')
        return []

    return _MMR(embdistrib, text_obj, candidates, X, beta, N, use_filtered)


def max_normalization(array):
    """
    Compute maximum normalization (max is set to 1) of the array
    :param array: 1-d array
    :return: 1-d array max- normalized : each value is multiplied by 1/max value
    """
    return 1/np.max(array) * array.squeeze(axis=1)


def get_aliases(kp_sim_between, candidates, threshold):
    """
    Find candidates which are very similar to the keyphrases (aliases)
    :param kp_sim_between: ndarray of shape (nb_kp , nb candidates) containing the similarity 已选关键词与候选词之间的相似性
    of each kp with all the candidates. Note that the similarity between the keyphrase and itself should be set to
    NaN or 0
    :param candidates: array of candidates (array of string)
    :return: list containing for each keyphrase a list that contain candidates which are aliases
    (very similar) (list of list of string)
    """

    kp_sim_between = np.nan_to_num(kp_sim_between, 0)  # 使用0代替nan元素
    idx_sorted = np.flip(np.argsort(kp_sim_between), 1)  # np.argsort 对相似度按照从小到大排序（1维上排序，即每个关键词与候选词的相似度排序），并返回索引值。# np.flip 将列的顺序进行颠倒，最后变成从大大小排序
    aliases = []
    for kp_idx, item in enumerate(idx_sorted):
        alias_for_item = []
        for i in item:
            if kp_sim_between[kp_idx, i] >= threshold:
                alias_for_item.append(candidates[i])
            else:
                break
        aliases.append(alias_for_item)

    return aliases
