# -*- coding: utf-8 -*-
# MorDL project: UPOS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from corpuscula.corpus_utils import _AbstractCorpus
import itertools
import junky
from mordl import WordEmbeddings
from mordl.base_tagger import BaseTagger
from mordl.upos_tagger_model import UposTaggerModel
from mordl.utils import LOG_FILE
import os
import sys


class UposTagger(BaseTagger):
    """"""

    def __init__(self):
        super().__init__()

    @classmethod
    def _prepare_corpus(cls, corpus, fields='UPOS', tags_to_remove=None):
        return super()._prepare_corpus(corpus, fields,
                                       tags_to_remove=tags_to_remove)

    def load(self, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
         super().load(UposTaggerModel, model_name, device=device,
                      dataset_device=dataset_device)

    def predict(self, corpus=None, batch_size=32, split=None):
        assert self._ds is not None, \
               "ERROR: the tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert self._model, \
               "ERROR: the tagger doesn't have a model. Call the train() " \
               'method first'
        if corpus is None:
            corpus = self._test_corpus
        elif isinstance(corpus, str):
            corpus = Conllu.load(corpus, **({'log_file': None} if silent else{}))
        elif (isinstance(corpus, type) and issubclass(corpus, _AbstractCorpus)) \
          or isinstance(corpus, _AbstractCorpus):
            corpus = corpus.test()
        elif callable(corpus):
            corpus = corpus()

        device = next(self._model.parameters()).device or junky.CPU
        def to_device(data):
            if isinstance(data, torch.Tensor):
                data = data.to(device)
            elif isinstance(data, Iterable):
                data = type(data)(to_device(x) for x in data)
            return data

        if not split:
            try:
                split = len(corpus)
            except TypeError:
                corpus = list(corpus)
                split = len(corpus)
        ds_y = self._ds.get_dataset('y')
        for start in itertools.count(step=split):
            try:
                corpus_ = corpus[start:start + split]
            except TypeError:
                corpus_ = []
                for i, sentence in enumerate(corpus, start=1):
                    corpus_.append(sentence)
                    if i == split:
                        break
            sentences, empties, nones = \
                junky.extract_conllu_fields(corpus_, fields=None,
                                            return_nones=True)
            self._transform_dataset(sentences)
            loader = self._ds.create_loader(batch_size=batch_size,
                                            shuffle=False)
            preds = []
            for batch in loader:
                batch = to_device(batch)
                with torch.no_grad():
                    pred = self._model(*batch)
                _, pred_indices = pred.max(2)
                preds.extend(pred_indices.cpu().numpy().tolist())
            values = ds_y.reconstruct(preds)
            for sentence in embed_conllu_fields(corpus, 'UPOS', values,
                                                empties=empties, nones=nones):
                yield sentence

    def evaluate(self, gold=None, test=None, batch_size=32, split=None,
                 silent=False):
        """Score the accuracy of the POS tagger against the *gold* standard.
        Remove POS tags from the *gold* standard text, retag it using the
        tagger, then compute the accuracy score. If *test* is not None, compute
        the accuracy of the *test* corpus with respect to the *gold*.

        :param gold: a corpus of tagged sentences to score the tagger on.
                     If *gold* is None then loaded test corpus is used
        :param test: a corpus of tagged sentences to compare with *gold*
        :param silent: suppress log
        :return: accuracy score of the tagger against the gold
        :rtype: float
        """
        n = c = 0
        if gold is None:
            gold = self._test_corpus
        elif isinstance(gold, str):
            gold = Conllu.load(gold, **({'log_file': None} if silent else{}))
        elif (isinstance(gold, type) and issubclass(gold, _AbstractCorpus)) \
          or isinstance(gold, _AbstractCorpus):
            gold = gold.test()
        elif callable(gold):
            gold = gold()
        if test is None:
            test = self.predict(test, batch_size=batch_size, split=split)
        elif isinstance(test, str):
            test = Conllu.load(test, **({'log_file': None} if silent else{}))
        elif (isinstance(test, type) and issubclass(test, _AbstractCorpus)) \
          or isinstance(test, _AbstractCorpus):
            test = test.test()
        elif callable(test):
            test = test()
        header = 'UPOS'
        if not silent:
            print('Evaluate ' + header, file=LOG_FILE)
        i = -1
        for i, (gold_sent, test_sent) in enumerate(zip(gold, test)):
            for j, (gold_token, test_token) in enumerate(zip(gold_sent, test_sent)):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_pos = gold_token['UPOS']
                    test_pos = test_token['UPOS']
                    n += 1
                    c += gold_pos == test_pos
        if not silent:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {}'.format(n), file=LOG_FILE)
                print(sp     + ' correct: {}'.format(c), file=LOG_FILE)
                print(sp     + '   wrong: {}'.format(n - c), file=LOG_FILE)
                print('Accuracy: {}'.format(c / n if n > 0 else 1.),
                      file=LOG_FILE)
        return c / n if n > 0 else 1.

    def train(self, model_name, device=None,
              epochs=sys.maxsize, min_epochs=0, bad_epochs=5,
              batch_size=32, control_metric='accuracy', max_grad_norm=None,
              tags_to_remove=None, word_emb_type=None, word_emb_path=None,
              word_emb_model_device=None, word_emb_tune_params=junky.kwargs(
                  name='bert-base-multilingual-cased', max_len=512,
                  epochs=4, batch_size=8
              ), word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=2, lstm_do=0,
              bn1=True, do1=.2, bn2=True, do2=.5, bn3=True, do3=.4, seed=None,
              log_file=LOG_FILE):

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if log_file:
            print('=== UPOS TAGGER TRAINING PIPELINE ===')

        model_fn, model_config_fn = self._get_filenames(model_name)[:2]

        # 1. Prepare corpora
        train, train_labels = self._prepare_corpus(
            self._train_corpus, tags_to_remove=tags_to_remove
        )
        test, test_labels = self._prepare_corpus(
            self._test_corpus, tags_to_remove=tags_to_remove
        ) if self._test_corpus is not None else (None, None)

        # 2. Tune embeddings
        def tune_word_emb(emb_type, emb_path, emb_model_device=None,
                          emb_tune_params=None):
            if emb_tune_params is True:
                emb_tune_params = {}
            elif isinstance(emb_tune_params, str):
                emb_tune_params = {'model_name': emb_tune_params}
            if isinstance(emb_tune_params, dict):
                if emb_type == 'bert':
                    if 'test_data' not in emb_tune_params and test:
                        emb_tune_params['test_data'] = (test, test_labels)
                    emb_tune_params['save_to'] = emb_path if emb_path else \
                                                 'upos_'
                    if emb_model_device and 'device' not in emb_tune_params:
                        emb_tune_params['device'] = emb_model_device
                    if 'seed' not in emb_tune_params:
                        emb_tune_params['seed'] = seed
                    if 'log_file' not in emb_tune_params:
                        emb_tune_params['log_file'] = log_file
                    if log_file:
                        print(file=log_file)
                    emb_path = WordEmbeddings.bert_tune(
                        train, train_labels, **emb_tune_params
                    )['model_name']
                else:
                    raise ValueError("ERROR: tune method for '{}' embeddings "
                                         .format(emb_type)
                                   + 'is not implemented')
            return emb_path

        word_emb_path = tune_word_emb(
            word_emb_type, word_emb_path,
            emb_model_device=word_emb_model_device,
            emb_tune_params=word_emb_tune_params
        )
        if word_next_emb_params:
            if isinstance(word_next_emb_params, dict):
                word_next_emb_params = [word_next_emb_params]
            for emb_params in word_next_emb_params:
                tune_params = emb_params.get('emb_tune_params',
                              emb_params.get('word_emb_tune_params'))
                emb_params['emb_path'] = tune_word_emb(
                    emb_params.get('emb_type', emb_params['word_emb_type']),
                    emb_params.get('emb_path', emb_params['word_emb_path']),
                    emb_model_device=emb_params.get('emb_model_device',
                                     emb_params.get('word_emb_model_device'),
                                     word_emb_model_device),
                    emb_tune_params=\
                        emb_params.get('emb_tune_params',
                        emb_params.get('word_emb_tune_params'))
                )

        if seed:
            junky.enforce_reproducibility(seed=seed)

        # 3. Create datasets
        if log_file:
            print('\nCREATE DATASETS', file=log_file)
        self._ds = self._create_dataset(
            train, word_emb_type=word_emb_type, word_emb_path=word_emb_path,
            word_emb_model_device=word_emb_model_device,
            word_transform_kwargs=word_transform_kwargs,
            word_next_emb_params=word_next_emb_params,
            with_chars=rnn_emb_dim or cnn_emb_dim, labels=train_labels)
        self._save_dataset(model_name)
        if test:
            ds_test = self._ds.clone(with_data=False)
            self._transform_dataset(test, labels=test_labels, ds=ds_test)
        else:
            ds_test = None

        # 4. Create model
        if log_file:
            print('\nCREATE A MODEL', file=log_file)
        self._model, criterion, optimizer, scheduler = \
            UposTaggerModel.create_model_for_train(
                len(self._ds.get_dataset('y').transform_dict),
                tags_pad_idx=self._ds.get_dataset('y').pad,
                vec_emb_dim=self._ds.get_dataset('x').vec_size
                                if word_emb_type is not None else
                            None,
                alphabet_size=len(self._ds.get_dataset('x_ch').transform_dict)
                                  if rnn_emb_dim or cnn_emb_dim else
                              0,
                char_pad_idx=self._ds.get_dataset('x_ch').pad
                                 if rnn_emb_dim or cnn_emb_dim else
                             0,
                rnn_emb_dim=rnn_emb_dim,
                cnn_emb_dim=cnn_emb_dim, cnn_kernels=cnn_kernels,
                emb_out_dim=emb_out_dim, lstm_hidden_dim=lstm_hidden_dim,
                lstm_layers=lstm_layers, lstm_do=lstm_do,
                bn1=True, do1=.2, bn2=True, do2=.5, bn3=True, do3=.4
            )
        if device:
            self._model = self._model.to(device)
        self._model.save_config(model_config_fn, log_file=log_file)

        # 5. Train model
        if log_file:
            print('\nTRAIN THE MODEL', file=log_file)
        def best_model_backup_method(model, model_score):
            if log_file:
                print('new maximum score {:.8f}'.format(model_score),
                      file=log_file)
            self._model.save_state_dict(model_fn, log_file=log_file)
        res_ = junky.train(
            None, self._model, criterion, optimizer, scheduler,
            best_model_backup_method, datasets=(self._ds, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric='accuracy',
            max_grad_norm=max_grad_norm,
            with_progress=log_file is not None, log_file=log_file
        )
        best_epoch, best_score = res_['best_epoch'], res_['best_score']
        res = {x: y for x, y in res_.items()
                        if x not in ['best_epoch', 'best_score']}

        # 6. Tune model
        if log_file:
            print('\nTUNE THE MODEL', file=log_file)
        self._model.load_state_dict(model_fn, log_file=log_file)
        criterion, optimizer, scheduler = self._model.adjust_model_for_tune()
        res_= junky.train(
            None, self._model, criterion, optimizer, scheduler,
            best_model_backup_method, datasets=(self._ds, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric='accuracy',
            max_grad_norm=max_grad_norm, best_score=best_score,
            with_progress=log_file is not None, log_file=log_file
        )
        if res_['best_epoch'] is not None:
            for key, value in res.items():
                if key == 'best_epoch':
                    res[key] += value
                elif key == 'best_score':
                    res[key] = value
                else:
                    res[key][:best_epoch] = value

        return res
