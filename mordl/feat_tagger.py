# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import junky
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class FeatTagger(BaseTagger):
    """"""

    def __init__(self, feat):
        if feat.find(':') == -1:
            feat = 'FEAT:' + feat
        self._feat = feat
        super().__init__()

    def load(self, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
         args, kwargs = junky.get_func_params(self.load, locals())
         super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
         args, kwargs = junky.get_func_params(self.predict, locals())
         return super().predict(self._feat, 'UPOS', *args, **kwargs)

    def evaluate(self, gold, test=None, label=None, batch_size=BATCH_SIZE,
                 split=None, clone_ds=False, log_file=LOG_FILE):
         args, kwargs = junky.get_func_params(self.evaluate, locals())
         return super().evaluate(self._feat, *args, **kwargs)

    def train(self, model_name,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=None, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
         args, kwargs = junky.get_func_params(self.train, locals())
         return super().train(self._feat, 'UPOS', FeatTaggerModel, 'upos',
                              *args, **kwargs)
