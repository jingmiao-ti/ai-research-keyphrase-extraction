# Copyright (c) 2017-present, Swisscom (Schweiz) AG.
# All rights reserved.
#
#Authors: Kamil Bennani-Smires, Yann Savary

# 此部分为论文第一步。对候选关键词进行过滤提取

"""Contain method that return list of candidate"""

import re

import nltk

# 正则表达式由一系列的词性标签组成，标签以尖括号为单位用来匹配一个词性对应的词。例如<NN>用于匹配句子中出现的名词，由于名词还有细分的如NNP,NNS等，可以用<NN.*>来表示所有名词的匹配。
# 匹配的英语语法块为  那些由零个或多个形容词后跟一个或多个名词组成的短语 {<名词 或 形容词><名词> }  名词名词 形容词名词

GRAMMAR_EN = """  NP:
        {<NN.*|JJ>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""

GRAMMAR_DE = """
NBAR:
        {<JJ|CARD>*<NN.*>+}  # [Adjective(s) or Article(s) or Posessive pronoun](optional) + Noun(s)
        {<NN>+<PPOSAT><JJ|CARD>*<NN.*>+}

NP:
{<NBAR><APPR|APPRART><ART>*<NBAR>}# Above, connected with APPR and APPART (beim vom)
{<NBAR>+}
"""

GRAMMAR_FR = """  NP:
        {<NN.*|JJ>*<NN.*>+<JJ>*}  # Adjective(s)(optional) + Noun(s) + Adjective(s)(optional)"""


def get_grammar(lang):
    if lang == 'en':
        grammar = GRAMMAR_EN
    elif lang == 'de':
        grammar = GRAMMAR_DE
    elif lang == 'fr':
        grammar = GRAMMAR_FR
    else:
        raise ValueError('Language not handled')
    return grammar


def extract_candidates(text_obj, no_subset=False):
    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :param lang: language (currently en, fr and de are supported)
    :return: list of candidate phrases (string)
    """
    
    # 这一部分为论文中的第一步，提取候选关键字
    keyphrase_candidate = set()
    # 基于词性的正则解析器，可以通过正则表达式匹配特定标记的词块。https://blog.csdn.net/zzulp/article/details/77414113
    np_parser = nltk.RegexpParser(get_grammar(text_obj.lang))  # Noun phrase parser
        
    # 按照定义的语法NP将句子解析为一颗树
    trees = np_parser.parse_sents(text_obj.pos_tagged)  # Generator with one tree per sentence
    
    # 提取中树中以NP为父节点的节点，也就是满足我们定义的语法规则正则表达式的节点
    for tree in trees:
        for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):  # For each nounphrase
            # Concatenate the token with a space
            # 候选词组之间用'，'隔开，每个候选词组之间的单词用空格隔开
            keyphrase_candidate.add(' '.join(word for word, tag in subtree.leaves()))  # keyphrase_candidate {'word1 word2','word1 word2 word3',```}
    
    # 只保留单词个数小于5的词组作为候选词组
    keyphrase_candidate = {kp for kp in keyphrase_candidate if len(kp.split()) <= 5}  

    if no_subset:
        keyphrase_candidate = unique_ngram_candidates(keyphrase_candidate)
    else:
        keyphrase_candidate = list(keyphrase_candidate)

    return keyphrase_candidate


def extract_sent_candidates(text_obj):
    """

    :param text_obj: input Text Representation see @InputTextObj
    :return: list of tokenized sentence (string) , each token is separated by a space in the string
    """
    return [(' '.join(word for word, tag in sent)) for sent in text_obj.pos_tagged]


def unique_ngram_candidates(strings):
    """
    ['machine learning', 'machine', 'backward induction', 'induction', 'start'] ->
    ['backward induction', 'start', 'machine learning']
    :param strings: List of string
    :return: List of string where no string is fully contained inside another string
    """
    results = []
    for s in sorted(set(strings), key=len, reverse=True):
        if not any(re.search(r'\b{}\b'.format(re.escape(s)), r) for r in results):
            results.append(s)
    return results
