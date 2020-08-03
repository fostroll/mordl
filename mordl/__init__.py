# -*- coding: utf-8 -*-
# MorDL project
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
MorDL provides tools for complete morphological sentence parsing and
named-entity recognition.
"""
from mordl._version import __version__
from mordl.word_embeddings import WordEmbeddings
from mordl.upos_tagger import UposTagger
from mordl.feat_tagger import FeatTagger
from mordl.feats_tagger import FeatsJointTagger as FeatsTagger
from mordl.lemma_tagger import LemmaTagger
from mordl.deprel_tagger import DeprelTagger, DeprelSeqTagger
from mordl.ne_tagger import NeTagger
from mordl.base_tagger import conll18_ud_eval
