#!/usr/bin/python
# -*- coding: utf-8 -*-
# MorDL project
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Example: A pipeline to train the MorDL model.
"""
from corpuscula.corpus_utils import download_ud, UniversalDependencies, \
                                    AdjustedForSpeech
from mordl import PosTagger, WordEmbeddings

BERT_MODEL_FN = 'bert-base-multilingual-cased'
MODEL_FN = 'pos_model'
SEED=42

# we use UD Taiga corpus only as example. For real model training comment
# Taiga and uncomment SynTagRus
#corpus_name = 'UD_Russian-Taiga'
corpus_name = 'UD_Russian-SynTagRus'

download_ud(corpus_name, overwrite=False)
train_corpus = dev_corpus = test_corpus = UniversalDependencies(corpus_name)
#train_corpus = dev_corpus = test_corpus = \
#                         AdjustedForSpeech(UniversalDependencies(corpus_name))


def get_model(load_corpuses=True, load_model=True):
    mp = MorphParser3(guess_pos=guess_pos, guess_lemma=guess_lemma,
                      guess_feat=guess_feat)
    if load_corpuses:
        mp.load_test_corpus(dev_corpus)
        mp.load_train_corpus(train_corpus)
    if load_model:
        mp.load(MODEL_FN)
    return mp

mp = get_model(load_model=False)
mp.parse_train_corpus()
mp._save_cdict('_cdict.pickle')
mp.save(MODEL_FN)

print()
mp = get_model()
mp.train_lemma(epochs=-3)
mp._save_lemma_model('_model.lemma.pickle')
mp.save(MODEL_FN)
print('== lemma ==')
mp.evaluate_lemma(test_corpus)

print()
mp = get_model()
mp.train_pos(rev=False, seed=SEED, epochs=-3)
mp._save_pos_model('_model.pos.pickle')
mp.save(MODEL_FN)
print('== pos 1 ==')
mp.evaluate_pos(test_corpus, rev=False)

print()
mp = get_model()
mp.train_pos(rev=True, seed=SEED, epochs=-3)
mp._save_pos_rev_model('_model.pos_rev.pickle')
mp.save(MODEL_FN)
print('== pos 1-rev ==')
mp.evaluate_pos(test_corpus, rev=True)

print()
mp = get_model()
mp.train_pos2(seed=SEED, epochs=-3, test_max_repeats=0)
mp._save_pos2_model('_model.pos2.pickle')
mp.save(MODEL_FN)
print('== pos 2 ==')
mp.evaluate_pos2(test_corpus, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== pos 2:{} =='.format(_r))
    mp.evaluate_pos2(test_corpus, with_backoff=False, max_repeats=_r)

print()
mp = get_model()
mp.train_feats(joint=False, rev=False, seed=SEED, epochs=-3)
mp._save_feats_models('_models.feats.pickle')
mp.save(MODEL_FN)
print('== feats 1s ==')
mp.evaluate_feats(test_corpus, joint=False, rev=False)

print()
mp = get_model()
mp.train_feats(joint=False, rev=True, seed=SEED, epochs=-3)
mp._save_feats_rev_models('_models.feats_rev.pickle')
mp.save(MODEL_FN)
print('== feats 1s-rev ==')
mp.evaluate_feats(test_corpus, joint=False, rev=True)

print()
mp = get_model()
mp.train_feats2(joint=False, seed=SEED, epochs=-3, test_max_repeats=0)
mp._save_feats2_models('_models.feats2.pickle')
mp.save(MODEL_FN)
print('== feats 2s ==')
mp.evaluate_feats2(test_corpus, joint=False, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== feats 2s:{} =='.format(_r))
    mp.evaluate_feats2(test_corpus, joint=False, with_backoff=False,
                       max_repeats=_r)

print()
mp = get_model()
mp.train_feats(joint=True, rev=False, seed=SEED, epochs=-3)
mp._save_feats_model('_model.feats.pickle')
mp.save(MODEL_FN)
print('== feats 1j ==')
mp.evaluate_feats(test_corpus, joint=True, rev=False)

print()
mp = get_model()
mp.train_feats(joint=True, rev=True, seed=SEED, epochs=-3)
mp._save_feats_rev_model('_model.feats_rev.pickle')
mp.save(MODEL_FN)
print('== feats 1j-rev ==')
mp.evaluate_feats(test_corpus, joint=True, rev=True)

print()
mp = get_model()
mp.train_feats2(joint=True, seed=SEED, epochs=-3, test_max_repeats=0)
mp._save_feats2_model('_model.feats2.pickle')
mp.save(MODEL_FN)
print('== feats 2j ==')
mp.evaluate_feats2(test_corpus, joint=True, with_backoff=True)
for _r in [0, 1, 2, 20]:
    print('== feats 2j:{} =='.format(_r))
    mp.evaluate_feats2(test_corpus, joint=True, with_backoff=False,
                       max_repeats=_r)

print()
mp = get_model()
for max_s in [None, 0, 1, 2]:
    for max_j in [None, 0, 1, 2]:
        print('== feats 3:{}:{} =='.format('' if max_s is None else max_s,
                                           '' if max_j is None else max_j))
        mp.evaluate_feats3(test_corpus,
                           with_s_backoff=max_s is None, max_s_repeats=max_s,
                           with_j_backoff=max_j is None, max_j_repeats=max_j)
