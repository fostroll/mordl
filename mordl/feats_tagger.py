# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import junky
from mordl import FeatTagger
from mordl.base_tagger import BaseTagger
from mordl.defaults import CONFIG_EXT, LOG_FILE, TRAIN_BATCH_SIZE


class FeatsTagger(BaseTagger):
    """"""

    def __init__(self):
        super().__init__()
        self._feats = {}

    def save(self, model_name, log_file=LOG_FILE):
        if not model_name.endswith(CONFIG_EXT):
            model_name += CONFIG_EXT
        with open(model_name, 'wt', encoding='utf-8') as f:
            print(json.dumps(self._feats, sort_keys=True, indent=4), file=f)
        if log_file:
            print('Config saved', file=log_file)

    def load(self, model_name,# device=None, dataset_device=None,
             log_file=LOG_FILE):
        if not model_name.endswith(CONFIG_EXT):
            model_name += CONFIG_EXT
        with open(model_name, 'rt', encoding='utf-8') as f:
            self._feats = json.loads(f.read())
        if log_file:
            print('### Load models for feats:', file=log_file)
        for feat, params in self._feats.items():
            if log_file:
                print('\n=== {}:'.format(feat), file=log_file)

    def train(self, model_name, feats=None,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path_suffix=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=None, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
        args, kwargs = get_func_params(FeatsTaggers.train, locals())
        del kwargs['feats']
        del kwargs['word_emb_path_suffix']

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if log_file:
            print('###### FEATS TAGGER TRAINING PIPELINE ######')
            print("\nWe're gonna train separate models for {} FEATS in train "
                      .format('the requested' if feats else 'all')
                + 'corpus. Feats are:\n')
        if not feats:
            feats = sorted(set(x for x in self._train_corpus
                                 for x in x
                                 for x in x['FEATS'].keys()))
        if log_file:
            print(', '.join(feats))

        for feat in feats:
            self._feats[feat] = ['{}-{}'.format(model_name, feat), device]

        res = {}
        for feat in feats:
            if log_file:
                print('\n--------- FEAT:{} ---------'.format(feat))

            tagger = FeatTagger(feat)
            tagger._train_corpus, tagger._test_corpus = \
                self._train_corpus, self._test_corpus
            if word_emb_path_suffix:
                kwargs['word_emb_path'] = \
                    'feat-{}_{}'.format(feat, word_emb_path_suffix)
            res[feat] = tagger.train(self._feats[feat][0], **kwargs)

            del tagger

        self.save(model_name, log_file=log_file)
        return res
