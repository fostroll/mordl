# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a named entity tagger class.
"""
from mordl import FeatTagger


class NeTagger(FeatTagger):
    """
    Named entity tagger class.
    """
    def __init__(self, cdict=None, feats_clip_coef=6, log_file=LOG_FILE):
        super().__init__('MISC:NE', cdict=cdict,
                         feats_clip_coef=feats_clip_coef if cdict else 0,
                         log_file=log_file)
