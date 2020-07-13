# -*- coding: utf-8 -*-
# MorDL project: FEATS:feat tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from junky import get_func_params
from mordl import UposTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class JointFeatsTagger(UposTagger):
    """"""

    def __init__(self, field='FEATS', work_field=None):
        super().__init__()
        self._orig_field = field
        self._field = work_field if work_field else field + '-joint'

    def load_train_corpus(self, *args, **kwargs):
        super().load_train_corpus(*args, **kwargs)
        [x.update({self._field:
                       '|'.join('='.join((y, x[self._orig_field][y]))
                                    for y in sorted(x[self._orig_field]))})
             for x in self._train_corpus for x in x]

    def load_test_corpus(self, *args, **kwargs):
        super().load_test_corpus(*args, **kwargs)
        [x.update({self._field:
                       '|'.join('='.join((y, x[self._orig_field][y]))
                                    for y in sorted(x[self._orig_field]))})
             for x in self._test_corpus for x in x]

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        args, kwargs = get_func_params(FeatTagger.predict, locals())
        return super().predict(self._feat, 'UPOS', *args, **kwargs)

    def evaluate(self, gold, test=None, label=None, batch_size=BATCH_SIZE,
                 split=None, clone_ds=False, log_file=LOG_FILE):
        args, kwargs = get_func_params(FeatTagger.evaluate, locals())
        return super().evaluate(self._feat, *args, **kwargs)

    def train(self, *args, **kwargs):
        key_vals = set(x[self._field] for x in self._train_corpus for x in x)
        [None if x[self._field] in key_vals else x.update({self._field: ''})
             for x in self._test_corpus for x in x]
        super().train(*args, **kwargs)
