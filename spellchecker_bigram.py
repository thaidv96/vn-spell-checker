from . import utils
import json
from pyvi import ViTokenizer
import re
import string


def score(prob, distance):
    return -prob / 100**distance


def gen_proposals(word, vocabs, telex_to_word):
    variants = utils.double_variants(word)
    res = variants.intersection(vocabs)
    return [telex_to_word[i] for i in res]


def load_model():
    with open("./spellchecker/cores/bigram_model.json") as f:
        temp = json.load(f)
        bigram_model = {}
        for k, v in temp.items():
            bigram_model[int(k)] = {int(i): j for i, j in v.items()}
    unigram_model = {k: sum(bigram_model[k].values()) for k in bigram_model}

    # normalize bigram:
    for k, v in bigram_model.items():
        for i, j in v.items():
            bigram_model[k][i] = j/unigram_model[k]
    # normalize unigram:
    total_tokens = sum(unigram_model.values())
    for k, v in unigram_model.items():
        unigram_model[k] = v/total_tokens
    return unigram_model, bigram_model


TELEX_DICT, VOCABS, WORD_TO_IDX, IDX_TO_WORD, WORD_TO_TELEX, TELEX_TO_WORD = utils.load_requirements()
del TELEX_TO_WORD['num']
TELEX_VOCABS = set(TELEX_TO_WORD.keys())
UNIGRAM_MODEL, BIGRAM_MODEL = load_model()


def predict(text):
    text = ViTokenizer.tokenize(text.strip())
    text = re.sub('_', ' ', ' '.join(text.split()))
    input_tokens = text.strip().split()
    input_tokens = [utils.word_to_telex(
        i, TELEX_DICT).strip() for i in input_tokens]
    BEAM_SIZE = 30
    # MAX_DEPTH = 5
    word = input_tokens[0]

    if len(input_tokens) == 1:
        if word in VOCABS:
            return word
    first_proposals = gen_proposals(word, TELEX_VOCABS, TELEX_TO_WORD)

    results = [([WORD_TO_IDX[w]], UNIGRAM_MODEL.get(WORD_TO_IDX[w], 0), utils.word_distance(word, utils.word_to_telex(w, TELEX_DICT)))
               for w in first_proposals]
    results = sorted(
        results, key=lambda x:  score(x[1], x[2]))[:BEAM_SIZE]
    i = 1
    for word in input_tokens[1:]:
        proposals = gen_proposals(word, TELEX_VOCABS, TELEX_TO_WORD)
        # Catch special case:
        # Not vietnamese word or just punctuation:
        if len(proposals) == 0 or word in string.punctuation:
            results = [(result + [word], u, v) for result, u, v in results]
            continue

        proposal_idx = [WORD_TO_IDX[w] for w in proposals]
        new_results = []
        for idx in proposal_idx:
            max_prob = 0
            new_max_sentence = None
            for sentence, prob, distance in results:
                new_sentence = sentence + [idx]
                new_distance = distance + \
                    utils.word_distance(word, utils.word_to_telex(
                        IDX_TO_WORD[idx], TELEX_DICT))
                if type(new_sentence[-2]) != int:
                    new_prob = prob * UNIGRAM_MODEL.get(idx, 0)
                else:
                    new_prob = prob * \
                        BIGRAM_MODEL.get(new_sentence[-2], {}).get(idx, 0)

                new_result = (new_sentence, new_prob, new_distance)
                if new_prob > 0:
                    new_results.append(new_result)
        new_results = sorted(
            new_results, key=lambda x: score(x[1], x[2]))
        results = new_results[:BEAM_SIZE]
        i += 1
    results = sorted(results, key=lambda x: score(x[1], x[2]))
    results = [(' '.join([IDX_TO_WORD.get(i, i)
                          for i in res]), distance, prob) for res, prob, distance in results]
    return results
