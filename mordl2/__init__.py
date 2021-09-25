# -*- coding: utf-8 -*-
# MorDL project
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
MorDL provides tools for complete morphological sentence parsing and
named-entity recognition.
"""
from mordl2._version import __version__
from mordl2.word_embeddings import WordEmbeddings
from mordl2.upos_tagger import UposTagger
from mordl2.feat_tagger import FeatTagger
from mordl2.feats_tagger import FeatsJointTagger as FeatsTagger
from mordl2.lemma_tagger import LemmaTagger
from mordl2.deprel_tagger import DeprelTagger, DeprelSeqTagger
from mordl2.ne_tagger import NeTagger
from mordl2.base_tagger import conll18_ud_eval

save_conllu = UposTagger.save_conllu
load_conllu = UposTagger.load_conllu
load_word_embeddings = WordEmbeddings.load
remove_rare_feats = UposTagger.remove_rare_feats
