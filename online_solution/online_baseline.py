import os


from functools import lru_cache
import argparse
import numpy as np
import pickle as pkl
import torch
from torch import softmax, sigmoid

from collections import Counter
from functools import partial

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pymystem3 import Mystem
import kenlm




TOKENIZATION_TYPE='sentencepiece'

ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


###SCORE FUNCTIONS:
TOXIC_CLASS=-1

def logits_to_toxic_probas(logits):
    if logits.shape[-1] > 1:
        activation = lambda x: softmax(x, -1)
    else:
        activation = sigmoid
    return activation(logits)[:, TOXIC_CLASS].cpu().detach().numpy()


def iterate_batches(data, batch_size=40):
    batch = []
    for x in data:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch

###!!!
def predict_toxicity(sentences, batch_size=5, threshold=0.5, return_scores=False, verbose=True):
    results = []
    for batch in iterate_batches(sentences, batch_size):
        normlized = [normalize(sent, max_tokens_per_word=5) for sent in batch]
        tokenized = tokenizer(normlized, return_tensors='pt', padding=True, max_length=512, truncation=True)
    
        logits = model(**{key: val for key, val in tokenized.items()}).logits
        preds = logits_to_toxic_probas(logits)
        if not return_scores:
            preds = preds >= threshold
        results.extend(preds)
    return results


def get_w2v_indicies(a):
    res = []
    if isinstance(a, str):
        a = a.split()
    for w in a:
        if w in embs_voc:
            res.append((w, embs_voc[w]))
        else:
            lemma = stemmer.lemmatize(w)[0]
            res.append((embs_voc.get(lemma), None))
    return res


def calc_embs(words):
    words = ' '.join(map(normalize, words))
    inds = get_w2v_indicies(words)
    return [(w, i if i is None else embs_vectors[i]) for w, i in inds]


def calc_single_embedding_dist(a, b):
    a_s, a_v = a  #слово, вектор
    b_s, b_v = b  #слово, вектор
    if a_s == b_s: #если слова равны
        return 0.0
    if a_v is None or b_v is None: #если одно из векторов нет - дать пизды 
        return 1.0
    a = a_v
    b = b_v
    # inexact match is punished by 0.1
    return 0.1 + 0.9 * (1 - a.dot(b) / np.linalg.norm(a) / np.linalg.norm(b)) / 2


def greedy_match_embs(a, b, max_dist=99999, cache=None, a_ind=0, b_ind=0):
    a_len = len(a) - a_ind #сколько слов мы еще не рассмотрели в а
    b_len = len(b) - b_ind #сколько слов мы еще не рассмотрели в b
    minlen = min(a_len, b_len) #минимальная длина строки
    maxlen = max(a_len, b_len) #минимальная длина строки
    if minlen == 0: 
        return np.minimum(maxlen, max_dist) 
    if maxlen - minlen >= max_dist: 
        return max_dist 
    
    if cache is None:
        cache = {}
    
    cache_key = (a_len, b_len)
    if cache_key in cache:
        return cache[cache_key]
        
    min_dist = max_dist
    
    first_dist = calc_single_embedding_dist(a[a_ind], b[b_ind])
    if max_dist >= first_dist:
        min_dist = np.minimum(min_dist, first_dist + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind + 1
        ))
    
    if first_dist > 0 and max_dist >= 1:
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind + 1, b_ind
        ))
        min_dist = np.minimum(min_dist, 1 + greedy_match_embs(
            a, b, max_dist, cache, a_ind, b_ind + 1
        ))
    
    cache[cache_key] = min_dist
    
    return min_dist


def calc_semantic_distance(a, b):
    a_embs = calc_embs(a)
    b_embs = calc_embs(b)
    
    clip_distance = 5  # this clips long computations
    return np.exp(-(greedy_match_embs(a_embs, b_embs, max_dist=clip_distance) / (0.6 * np.log(1 + len(a)))) ** 2)


def distance_score(original, fixed):
    original = original.split()
    fixed = fixed.split()
    
    return calc_semantic_distance(original, fixed)


def compute_lmdiff(original, fixed):
    original_lm_logproba = lm.score(original, bos=True, eos=True)
    fixed_lm_logproba = lm.score(fixed, bos=True, eos=True)
    
    probability_fraction = 10**((fixed_lm_logproba - original_lm_logproba) / 25)
    
    return np.clip(probability_fraction, 0.0, 1.0)


def compute_score(original_sentences, fixed_sentences, threshold=0.5, batch_size=5):
    fixed_toxicities = predict_toxicity(fixed_sentences, threshold=threshold, batch_size=batch_size)
    scores = []
    lmdiffs = []
    emb_dists = []
    for original_sentence, fixed_sentence, fixed_toxicity in zip(
        original_sentences, fixed_sentences, fixed_toxicities
    ):
        original_sentence = normalize(original_sentence)
        fixed_sentence = normalize(fixed_sentence)
        
        distance = distance_score(original_sentence, fixed_sentence)
        lmdiff = compute_lmdiff(original_sentence, fixed_sentence)
        
        score = (1 - fixed_toxicity) * distance * lmdiff
        
        scores.append(score)

    return np.mean(scores)


def new_checker(original_texts, fixed_texts):
    
    original_texts = list(map(str.strip, original_texts))
    fixed_texts = list(map(str.strip, fixed_texts))
    
    assert len(original_texts) == len(fixed_texts)
    
    with torch.inference_mode(True):
        return (100 * compute_score(original_texts, fixed_texts))
###


def is_word_start(token):
    if TOKENIZATION_TYPE == 'sentencepiece':
        return token.startswith('▁')
    if TOKENIZATION_TYPE == 'bert':
        return not token.startswith('##')
    raise ValueError("Unknown tokenization type")



def normalize(sentence, max_tokens_per_word=20):
    def validate_char(c):
        return c in ALLOWED_ALPHABET

    sentence = ''.join(map(lambda c: c if validate_char(c) else ' ', sentence.lower()))
    ids = tokenizer(sentence)['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)[1:-1]

    result = []
    num_continuation_tokens = 0
    for token in tokens:
        if not is_word_start(token):
            num_continuation_tokens += 1
            if num_continuation_tokens < max_tokens_per_word:
                result.append(token.lstrip('#▁'))
        else:
            num_continuation_tokens = 0
            result.extend([' ', token.lstrip('▁#')])

    return ''.join(result).strip()








def load_embeddings(path):
    embs_file = np.load(path, allow_pickle=True)
    embs_vectors = embs_file['vectors']
    embs_voc = embs_file['voc'].item()

    return embs_voc, embs_vectors









def sort_by_toxicity(words):
    #toxicities = [toxicity.get(w) if w in toxicity else 0.5 for w in words]
    toxicities = [toxicity.get(w) if w in toxicity else predict_toxicity([w], return_scores = True)[0] for w in words]

    result = [[toxicities[i], i, words[i]] for i in range(len(words))]
    result.sort()
    return result


def detox_light(line):

    
    words = normalize(line).split()
    sorted_words = sort_by_toxicity(words)
  
    while sorted_words:
        toxic_toxicity, toxic_idx, toxic_word = sorted_words.pop()
        if toxic_toxicity > 0.77:
            words[toxic_idx] = "Спасибо"

        

    
    return ' '.join(words)


def detox_tough(line):
    words = normalize(line).split()
    sorted_words = sort_by_toxicity(words)

    score = new_checker([line], [' '.join(words)])
    
    while sorted_words:
        toxic_toxicity, toxic_idx, toxic_word = sorted_words.pop()
        
        fixed_word = "спасибо"
        
        

        words[toxic_idx] = fixed_word
        new_score = new_checker([line], [' '.join(words)])

        if new_score >= score:
            score = new_score
        else:
            words[toxic_idx] = toxic_word
            break
    
    return ' '.join(words)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('original_texts', type=argparse.FileType('r'))
    parser.add_argument('fixed_texts', type=argparse.FileType('w'))
    parser.add_argument('--data', type=argparse.FileType('rb'), required=True)
    parser.add_argument('--embeddings', type=str, required=True)
    parser.add_argument('--tokenizer', required=True, type=str)
    parser.add_argument('--lm_model', required=True, type=str)

    return parser.parse_args()




if __name__ == '__main__':
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(args.tokenizer)
    
    lm = kenlm.Model(args.lm_model)
    
    # load precomputed toxicities
    with args.data as f:
        toxicity = pkl.load(f)
        
        
    # load embeddings
    embs_voc, embs_vectors = load_embeddings(args.embeddings)
    
    # initialize stemer
    stemmer = Mystem()
    
    with args.original_texts, args.fixed_texts:
        for line in tqdm(args.original_texts):
            print(detox_light(line.strip()), file=args.fixed_texts)
