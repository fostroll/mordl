# -*- coding: utf-8 -*-
# MorDL project: FEATS:feat tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a single-tag FEAT tagger class.
"""
from mordl import FeatTagger


class AtomicTagger(FeatTagger):
    """
    A class of single-feature tagger.

    Args:

    **feat** (`str`): the name of the feat which needs to be predicted by the
    tagger. May contains prefix, separated by a colon (`:`). In that case, the
    prefix treat as a field name. Otherwise, we get `'FEATS'` as a field name.
    Examples: `'Animacy'`; `'MISC:NE'`.

    **feats_prune_coef** (`int`): feature prunning coefficient which allows to
    eliminate all features that have a low frequency. For each UPOS tag, we
    get a number of occurences of the most frequent feature from FEATS field,
    divide it by **feats_prune_coef** and use only those features, number of
    occurences of which is greater than that value, to improve the prediction
    quality.
    * `feats_prune_coef=0` means "do not use feats";
    * `feats_prune_coef=None` means "use all feats";
    * default `feats_prune_coef=6`.

    **NB**: the argument is relevant only if **feat** is not from FEATS field.
    """
    def __init__(self, field, feats_prune_coef=6):
        super().__init__(':', feats_prune_coef=feats_prune_coef)
        self._field = field
