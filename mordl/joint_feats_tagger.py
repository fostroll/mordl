# -*- coding: utf-8 -*-
# MorDL project: FEATS:feat tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class JointFeatsTagger(BaseTagger):
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

    def load(self, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
        args, kwargs = get_func_params(JointFeatsTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(JointFeatsTagger.predict, locals())
        kwargs['save_to'] = None

        def process(corpus):
            for sentence in corpus:
                [x.update({self._orig_field: OrderedDict(
                    {x: y for x, y in [x.split('=')
                          for x in x[self._orig_field].split('|')]}
                )}) for x in (sentence[0] if with_orig else sentence)]
                yield sentence

        corpus = process(
            super().predict(self._orig_field, 'UPOS', *args, **kwargs)
        )
        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, gold, test=None, feat=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        assert not feat and label, \
            "ERROR: To evaluate the exact label you must specify it's " \
            'feat, too'
        args, kwargs = get_func_params(JointFeatsTagger.evaluate, locals())
        del kwargs['feat']
        field = self._orig_field
        if feat:
            field += ':' + feat
        return super().evaluate(field, *args, **kwargs)

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
        key_vals = set(x[self._field] for x in self._train_corpus for x in x)
        [None if x[self._field] in key_vals else x.update({self._field: ''})
             for x in self._test_corpus for x in x]
        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)
