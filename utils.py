import os
from pyxdameraulevenshtein import damerau_levenshtein_distance
import re
import json


def variants(word, alphabet):
    """get all possible variants for a word with 1 edit-distance"""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)


def double_variants(word):
    res = set()
    alphabet = set('abcdefghijklmnopqrstuvwxyz')
    for w in variants(word, alphabet):
        res.update(variants(w, alphabet))
    return res


def normalize_telex(telex):
    """
    Chuẩn hoá telex: đưa thanh điệu và dấu (w) về cuối
    """
    set_consonants = set("bcdđfghjklmnpqrstvwxz")
    first_vowel_idx = -1
    for idx, c in enumerate(telex):
        if c not in set_consonants:
            first_vowel_idx = idx
            break
    for accent in 'sfrxj':
        if accent in telex[first_vowel_idx:-1]:
            telex = telex[: telex.rindex(accent)] + \
                telex[telex.rindex(accent) + 1:] + accent
            break
    if 'w' in telex:
        telex = re.sub("w", '', telex) + 'w'
    return telex


def word_to_telex(word, telex_dict):
    """
    Chuyển đổi tiếng việt có dấu thành cách gõ telex
    """
    consonants = "bcdghklmnpqrstx"
    res = ""

    for c in word.lower():
        if c in consonants:
            res += c
        else:
            converted_c = telex_dict['typical'].get(c, c)
            res += converted_c
    accent = re.search("\[\w\]", res)
    if accent:
        accent = accent[0]
        res = re.sub("\[\w\]", '', res) + accent[1]

    return normalize_telex(res)


def word_distance(word1, word2):
    """
    Tính khoảng cách edit giữa 2 từ 
    """
    return damerau_levenshtein_distance(word1, word2)

    # return Levenshtein.distance(word1, word2)


def load_requirements():
    print(os.getcwd())
    with open("./spellchecker/cores/telex_dict.json", encoding='utf8') as f:
        TELEX_DICT = json.load(f)
    with open("./spellchecker/cores/vocabs.txt") as f:
        VOCABS = f.read().strip().split("\n")

    WORD2IDX = {k: v for v, k in enumerate(VOCABS)}
    IDX2WORD = {k: v for k, v in enumerate(VOCABS)}
    WORD_TO_TELEX = {w: word_to_telex(w, TELEX_DICT) for w in VOCABS}
    TELEX_TO_WORD = {v: k for k, v in WORD_TO_TELEX.items()}
    return TELEX_DICT, set(VOCABS), WORD2IDX, IDX2WORD, WORD_TO_TELEX, TELEX_TO_WORD
