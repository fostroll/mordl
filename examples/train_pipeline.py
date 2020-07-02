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
from mordl import PosTagger

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

pt = PosTagger()
#pt.load_train_corpus(train_corpus)
#pt.load_test_corpus(test_corpus)

pt.train(MODEL_FN, model_config_file=True, device='cuda:11',
         word_emb_type='bert', word_emb_path=None,
         word_emb_model_device='cuda:6', word_emb_tune_params=True,
         rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
         seed=SEED)
