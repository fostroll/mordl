# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import json
from junky import clear_tqdm, get_func_params
from mordl import FeatTagger
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, CONFIG_EXT, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class FeatsTagger(BaseTagger):
    """"""

    def __init__(self, field='FEATS'):
        super().__init__()
        self._field = field
        self._feats = {}

    def save(self, model_name, log_file=LOG_FILE):
        if not model_name.endswith(CONFIG_EXT):
            model_name += CONFIG_EXT
        with open(model_name, 'wt', encoding='utf-8') as f:
            print(json.dumps({x: y[0] if isinstance(y, dict) else y
                                  for x, y in self._feats.items()},
                             sort_keys=True, indent=4), file=f)
        if log_file:
            print('Config saved', file=log_file)

    def load(self, model_name, log_file=LOG_FILE):
        if not model_name.endswith(CONFIG_EXT):
            model_name += CONFIG_EXT
        with open(model_name, 'rt', encoding='utf-8') as f:
            self._feats = json.loads(f.read())
        if log_file:
            print('### Load models for feats:', file=log_file)
        for feat, model_name_ in self._feats.items():
            if log_file:
                print('\n--- {}:'.format(feat), file=log_file)
            tagger = FeatTagger(feat)
            tagger.load(model_name_)
            self._feats[feat] = [model_name, tagger]
        if log_file:
            print('### done.', file=log_file)

    def predict(self, corpus, feat=None, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE):

        args, kwargs = get_func_params(FeatsTagger.predict, locals())
        del kwargs['feat']

        if feat:
            attrs = self._feats[feat]
            tagger = attrs[1] if isinstance(attrs, list) > 1 else None
            assert isinstance(tagger, FeatTagger), \
                'ERROR: Model is not loaded. Use the .load() method prior'
            corpus = tagger.predict(*args, **kwargs)

        else:
            kwargs['with_orig'] = False
            kwargs['save_to'] = None

            def process(corpus):
                corpus = self._get_corpus(corpus, asis=True,
                                          log_file=log_file)
                for start in itertools.count(step=split if split else 1):
                    if isinstance(corpus, Iterator):
                        if split:
                            corpus_ = []
                            for i, sentence in enumerate(corpus, start=1):
                                corpus_.append(sentence)
                                if i == split:
                                    break
                        else:
                            corpus_ = list(corpus)
                    else:
                        if split:
                            corpus_ = corpus[start:start + split]
                        else:
                            corpus_ = corpus
                    if not corpus_:
                        break

                    res_corpus_ = deepcopy(corpus_) if with_orig else corpus_

                    for attrs in self._feats.values():
                        tagger = attrs[1] \
                                     if isinstance(attrs, list) > 1 else \
                                 None
                        assert isinstance(tagger, FeatTagger), \
                            'ERROR: Model is not loaded. Use the .load() ' \
                            'method prior'
                        corpus = tagger.predict(res_corpus_, **kwargs)

                    if with_orig:
                        for orig_sentence, sentence in zip(corpus_,
                                                           res_corpus_):
                            yield sentence, orig_sentence
                    else:
                        for sentence in res_corpus_:
                            yield sentence

            corpus = process(corpus)
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
        args, kwargs = get_func_params(FeatTagger.evaluate, locals())
        del kwargs['feat']
        field = self._field
        if feat:
            field += ':' + feat
        return super().evaluate(field, *args, **kwargs)

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
        args, kwargs = get_func_params(FeatsTagger.train, locals())
        del kwargs['feats']
        del kwargs['word_emb_path_suffix']

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if log_file:
            print('###### {} TAGGER TRAINING PIPELINE ######'
                      .format(self._field), file=log_file)
            print("\nWe're gonna train separate models for {} {} in train "
                      .format('the requested' if feats else 'all',
                              self._field)
                + 'corpus. Feats are:\n', file=log_file)
        if not feats:
            feats = sorted(set(x for x in self._train_corpus
                                 for x in x
                                 for x in x[self._field].keys()))
        if log_file:
            print(', '.join(feats), file=log_file)

        res = {}
        for feat in feats:
            if log_file:
                print(file=log_file)
                clear_tqdm()

            model_name_ = '{}-{}'.format(model_name, feat.lower())
            self._feats[feat] = model_name_

            tagger = FeatTagger(self._field + ':' + feat)
            tagger._train_corpus, tagger._test_corpus = \
                self._train_corpus, self._test_corpus
            if word_emb_path_suffix:
                kwargs['word_emb_path'] = \
                    '{}-{}_{}'.format(self._field.lower(), feat.lower(),
                                      word_emb_path_suffix)
            res[feat] = tagger.train(model_name_, **kwargs)

            del tagger

        self.save(model_name, log_file=log_file)
        if log_file:
            print('\n###### {} TAGGER TRAINING HAS FINISHED ######\n'
                      .format(self._field), file=log_file)
            print(("Now, check the separate {} models' and datasets' "
                   'config files and consider to change some device names '
                   'to be able load all the models jointly. You can find '
                   'the separate models\' list in the "{}" config file. '
                   "Then, use the `.load('{}')` method to start working "
                   'with the {} tagger.').format(self._field, model_name,
                                                 model_name, self._field),
                      file=log_file)
        return res


class FeatsJointTagger(BaseTagger):
    """"""

    def __init__(self, field='FEATS', work_field=None):
        super().__init__()
        self._orig_field = field
        self._field = work_field if work_field else field + 'j'

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
        args, kwargs = get_func_params(FeatsJointTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(FeatsJointTagger.predict, locals())
        kwargs['save_to'] = None

        def process(corpus):
            for sentence in corpus:
                for token in (sentence[0] if with_orig else sentence):
                    token[self._orig_field] = OrderedDict(
                        [(x, y) for x, y in
                             [x.split('=')
                                  for x in x[self._field].split('|')]]
                    )
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
        args, kwargs = get_func_params(FeatsJointTagger.evaluate, locals())
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
        args, kwargs = get_func_params(FeatsJointTagger.train, locals())
        key_vals = set(x[self._field] for x in self._train_corpus for x in x)
        [None if x[self._field] in key_vals else x.update({self._field: ''})
             for x in self._test_corpus for x in x]
        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)
