# -*- coding: utf-8 -*-
# MorDL project: MISC:NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a Named-entity tagger class.
"""
from mordl import FeatTagger


class NeTagger(FeatTagger):
    """
    The named-entity tagger class. We use the feature 'NE' of MISC field as
    the place where Named-entity tags are stored.

    Args:

    **feats_prune_coef** (`int`): feature prunning coefficient which allows to
    eliminate all features that have a low frequency. For each UPOS tag, we
    get a number of occurences of the most frequent feature from FEATS field,
    divide it by **feats_prune_coef** and use only those features, number of
    occurences of which is greater than that value, to improve the prediction
    quality.
    * `feats_prune_coef=0` means "do not use feats";
    * `feats_prune_coef=None` means "use all feats";
    * default `feats_prune_coef=6`.

    **embs**: `dict` with paths to the embeddings file as keys and
    corresponding embeddings models as values. If tagger needs to load any
    embeddings model, firstly, model is looked up it in that `dict`.

    During init, **embs** is copied to the `embs` attribute of the creating
    object, and this attribute may be used further to share already loaded
    embeddings with another taggers.
    """
    def __init__(self, feats_prune_coef=6, embs=None):
        super().__init__('MISC:NE', feats_prune_coef=feats_prune_coef,
                         embs=None)
