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


class FeatsTagger(BaseTagger):
    """"""

    def __init__(self):
        super().__init__()
        self._feats = {}

    def save(self, model_name, log_file=LOG_FILE):
        if not model_name.endswith(CONFIG_EXT):
            model_name += CONFIG_EXT
        with open(model_name, 'wt', encoding='utf-8') as f:
            config = {x: [y[0], next(y[1].parameters()).device
                                    if isinstance(y[1], FeatTagger) else
                                y[1]]
                             if isinstance(y[1], dict) else
                         y
                          for x, y in self._feats.items()}
            print(json.dumps(config, sort_keys=True, indent=4), file=f)
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

    def predict(self, corpus, feat=None, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE):

        args, kwargs = get_func_params(FeatsTagger.predict, locals())
        del kwargs['feat']

        if feat:
            attrs = self._feats[feat]
            tagger = attrs[1] if isinstance(attrs) > 1 else None
            assert isinstance(tagger, FeatTagger), \
                'ERROR: model is not loaded. Use the .load() method prior'
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
                        tagger = attrs[1] if isinstance(attrs) > 1 else None
                        assert isinstance(tagger, FeatTagger), \
                            'ERROR: model is not loaded. Use the .load() ' \
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
            "ERROR: to evaluate the exact label you must specify it's " \
            'feat, too'
        args, kwargs = get_func_params(FeatTagger.evaluate, locals())
        del kwargs['feat']
        field = 'FEATS'
        if feat:
            field += ':{}' + feat
        return super().evaluate(field, *args, **kwargs)

    def evaluate(self, gold, test=None, feat=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):

        if feat:
            attrs = self._feats[feat]
            tagger = attrs[1] if isinstance(attrs) > 1 else None
            assert isinstance(tagger, FeatTagger), \
                'ERROR: model is not loaded. Use the .load() method prior'
            args, kwargs = get_func_params(FeatsTagger.evaluate, locals())
            del kwargs['feat']
            res = tagger.evaluate(*args, **kwargs)

        else:
            assert not label, \
                "ERROR: to evaluate the exact label you must specify it's " \
                'feat, too'

            gold = self._get_corpus(gold, log_file=log_file)
            corpora = zip(gold, self._get_corpus(test, log_file=log_file)) \
                          if test else \
                      self.predict(gold, with_orig=True,
                                   batch_size=batch_size, split=split,
                                   clone_ds=clone_ds, log_file=log_file)
            field_ = field.split(':')
            field = field_[0]
            name = field_[1] if len(field_) > 1 else None
            header = ':'.join(field_[:2])
            if label:
                header += '::' + label
            if log_file:
                print('Evaluate ' + header, file=log_file)
            n = c = nt = ct = ca = ce = cr = 0
            i = -1
            for i, sentences in enumerate(corpora):
                for gold_token, test_token in zip(*sentences):
                    wform = gold_token['FORM']
                    if wform and '-' not in gold_token['ID']:
                        gold_label = gold_token[field]
                        test_label = test_token[field]
                        if name:
                            gold_label = gold_label.get(name)
                            test_label = test_label.get(name)
                        n += 1
                        if (label and (gold_label == label
                                    or test_label == label)) \
                        or (not label and (gold_label or test_label)):
                            nt += 1
                            if gold_label == test_label:
                                c += 1
                                ct += 1
                            elif not gold_label or (label
                                                and gold_label != label):
                                ce += 1
                            elif not test_label or (label
                                                and test_label != label):
                                ca += 1
                            else:
                                cr += 1
                        else:
                            c += 1
            if log_file:
                if i < 0:
                    print('Nothing to do!', file=log_file)
                else:
                    sp = ' ' * (len(header) - 2)
                    print(header + ' total: {}'.format(nt), file=log_file)
                    print(sp   + ' correct: {}'.format(ct), file=log_file)
                    print(sp   + '   wrong: {}{}'.format(
                        nt - ct, ' [{} excess / {} absent{}]'.format(
                            ce, ca, '' if label else ' / {} wrong type'.format(cr)
                        ) if nt != n else ''
                    ), file=log_file)
                    print(sp   + 'Accuracy: {}'.format(ct / nt if nt > 0 else 1.),
                          file=log_file)
                    if nt != n:
                        print('[Total accuracy: {}]'
                                  .format(c / n if n > 0 else 1.), file=log_file)
            res = ct / nt if nt > 0 else 1.

        return res

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
            print('###### FEATS TAGGER TRAINING PIPELINE ######',
                  file=log_file)
            print("\nWe're gonna train separate models for {} FEATS in train "
                      .format('the requested' if feats else 'all')
                + 'corpus. Feats are:\n', file=log_file)
        if not feats:
            feats = sorted(set(x for x in self._train_corpus
                                 for x in x
                                 for x in x['FEATS'].keys()))
        if log_file:
            print(', '.join(feats), file=log_file)

        res = {}
        for feat in feats:
            if log_file:
                print(file=log_file)
                clear_tqdm()

            model_name_ = '{}-{}'.format(model_name, feat.lower())
            self._feats[feat] = [model_name_, str(device)] if device else \
                                model_name_

            tagger = FeatTagger(feat)
            tagger._train_corpus, tagger._test_corpus = \
                self._train_corpus, self._test_corpus
            if word_emb_path_suffix:
                kwargs['word_emb_path'] = \
                    'feats-{}_{}'.format(feat.lower(), word_emb_path_suffix)
            res[feat] = tagger.train(model_name_, **kwargs)

            del tagger

        self.save(model_name, log_file=log_file)
        return res
