# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a named entity tagger class.
"""
from mordl import FeatTagger
from mordl.defaults import LOG_FILE


class NeTagger(FeatTagger):
    """
    Named entity tagger class.
    """
    def __init__(self, feats_prune_coef=6):
        super().__init__('MISC:NE', feats_prune_coef=feats_prune_coef)
