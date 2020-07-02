# -*- coding: utf-8 -*-
# MorDL project: Base tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from morra.base_parser import BaseParser


class BaseTagger(BaseParser):
    """"""
    def __err_hideattr(obj, name):
        raise AttributeError("'{}' {} '{}'".format(
            obj.__name__ if isinstance(obj, type) else obj.__class__.__name__,
            'object has no attribute',
            name
        ))
    parse_train_corpus = \
        property(lambda self: self.__err_hideattr('parse_train_corpus'))
    _train_init = \
        property(lambda self: self.__err_hideattr(self, '_train_init'))
    _train_eval = \
        property(lambda self: self.__err_hideattr(self, '_train_eval'))
    _train_done = \
        property(lambda self: self.__err_hideattr(self, '_train_done'))

    def __init__(self):
        super().__init__()
        delattr(self, '_cdict')

    def backup(self):
        pass

    def restore(self):
        pass

    def save(self):
        pass

    def load(self):
        pass

    def load_train_corpus(self, corpus, append=False, test=None, seed=None):
        super().load_train_corpus(corpus, append=append, parse=False,
                                  test=test, seed=seed)
