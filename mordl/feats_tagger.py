# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from copy import deepcopy
import itertools
import junky
from mordl import WordEmbeddings
from mordl.base_tagger import BaseTagger
from mordl.ne_tagger_model import NeTaggerModel
from mordl.utils import LOG_FILE
import os
import torch
from typing import Iterator
import sys


class FeatsTagger(BaseTagger):
    """"""

    def __init__(self):
        super().__init__()

    def load(self, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
         args, kwargs = junky.get_func_params(self.load, locals())
         super().load(NeTaggerModel, *args, **kwargs)

    def predict(self, corpus, batch_size=32, split=None, with_orig=False,
                save_to=None, log_file=LOG_FILE):
        assert self._ds is not None, \
               "ERROR: the tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert self._model, \
               "ERROR: the tagger doesn't have a model. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'

        def process(corpus):
            corpus = self._get_corpus(corpus, asis=True, log_file=log_file)
            device = next(self._model.parameters()).device or junky.CPU

            ds = self._ds.clone()
            ds_y = self._ds.get_dataset('y')
            ds.remove('y')

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
                res = \
                    junky.extract_conllu_fields(
                        corpus_, fields='UPOS', with_empty=True, return_nones=True
                    )
                sentences, tags, empties, nones = \
                    res[0], res[1:-2], res[-2], res[-1]
                self._transform_dataset(sentences, tags=tags, ds=ds)
                loader = ds.create_loader(batch_size=batch_size, shuffle=False)
                preds = []
                for batch in loader:
                    batch = junky.to_device(batch, device)
                    with torch.no_grad():
                        pred = self._model(*batch)
                    _, pred_indices = pred.max(2)
                    preds.extend(pred_indices.cpu().numpy().tolist())
                values = ds_y.reconstruct(preds)
                if with_orig:
                    res_corpus_ = deepcopy(corpus_)
                    for orig_sentence, sentence in zip(
                        corpus_, junky.embed_conllu_fields(
                            res_corpus_, 'MISC:NE:_', values,
                            empties=empties, nones=nones
                        )
                    ):
                        yield sentence, orig_sentence
                else:
                    for sentence in junky.embed_conllu_fields(
                        corpus_, 'MISC:NE:_', values,
                        empties=empties, nones=nones
                    ):
                        yield sentence

        corpus = process(corpus)
        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, gold, test=None, ne=None, batch_size=32, split=None,
                 log_file=LOG_FILE):

        gold = self._get_corpus(gold, log_file=log_file)
        corpora = zip(gold, self._get_corpus(test, log_file=log_file)) \
                      if test else \
                  self.predict(corpus=gold, batch_size=batch_size,
                               split=split, with_orig=True, log_file=log_file)
        header = 'NE'
        if log_file:
            print('Evaluate ' + header, file=LOG_FILE)
        n = c = nt = ct = ca = ce = cr = 0
        i = -1
        for i, sentences in enumerate(corpora):
            for gold_token, test_token in zip(*sentences):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_ne = gold_token['MISC'].get('NE')
                    test_ne = test_token['MISC'].get('NE')
                    n += 1
                    if (ne and (gold_ne == ne or test_ne == ne)) \
                    or (not ne and (gold_ne or test_ne)):
                        nt += 1
                        if gold_ne == test_ne:
                            c += 1
                            ct += 1
                        elif not gold_ne or (ne and gold_ne != ne):
                            ce += 1
                        elif not test_ne or (ne and test_ne != ne):
                            ca += 1
                        else:
                            cr += 1
                    else:
                        c += 1
        if log_file:
            if i < 0:
                print('Nothing to do!', file=LOG_FILE)
            else:
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {}'.format(nt), file=LOG_FILE)
                print(sp   + ' correct: {}'.format(ct), file=LOG_FILE)
                print(sp   + '   wrong: {} [{} excess / {} absent{}]'
                                 .format(nt - ct, ce, ca, '' if ne else
                                                          ' / {} wrong type'
                                                              .format(cr)),
                      file=LOG_FILE)
                print(sp   + 'Accuracy: {}'.format(ct / nt if nt > 0 else 1.))
                print('[Total accuracy: {}]'.format(c / n if n > 0 else 1.))
        return ct / nt if nt > 0 else 1.

    def train(self, model_name, device=None,
              epochs=sys.maxsize, min_epochs=0, bad_epochs=5,
              batch_size=32, control_metric='accuracy', max_grad_norm=None,
              tags_to_remove=None, word_emb_type=None, word_emb_path=None,
              word_emb_model_device=None, word_emb_tune_params=junky.kwargs(
                  name='bert-base-multilingual-cased', max_len=512,
                  epochs=4, batch_size=8
              ), word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=None, num_upos=0, upos_pad_idx=0, emb_out_dim=512,
              lstm_hidden_dim=256, lstm_layers=2, lstm_do=0, bn1=True, do1=.2,
              bn2=True, do2=.5, bn3=True, do3=.4, seed=None, log_file=LOG_FILE):

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if log_file:
            print('=== FEATS TAGGER TRAINING PIPELINE ===')

        feats = sorted(set(x for x in self._train_corpus
                             for x in x
                             for x in x['FEATS'].keys()))
        if log_file:
            print("\nWe're gonna train separate models for all FEATS in train "
                  'corpus. Feats are:\n')
            print(', '.join(feats))

        total_res = {}
        for feat in feats:
            if log_file:
                print('\n--------- FEAT:{} ---------'.format(feat))

            model_fn, model_config_fn = \
                self._get_filenames(model_name + '_' + feat)[:2]

            # 1. Prepare corpora
            train = self._prepare_corpus(
                self._train_corpus, fields=['UPOS', 'FEATS:' + feat + ':_'],
                tags_to_remove=tags_to_remove
            )
            test = self._prepare_corpus(
                self._test_corpus, fields=['UPOS', 'FEATS:' + feat + ':_'],
                tags_to_remove=tags_to_remove
            ) if self._test_corpus is not None else [None]

            # 2. Tune embeddings
            def tune_word_emb(emb_type, emb_path, emb_model_device=None,
                              emb_tune_params=None):
                emb_path_prefix = 'feat_' + feat + '_'
                if emb_path:
                    emb_path = emb_path_prefix + emb_path
                if emb_tune_params is True:
                    emb_tune_params = {}
                elif isinstance(emb_tune_params, str):
                    emb_tune_params = {'model_name': emb_tune_params}
                if isinstance(emb_tune_params, dict):
                    if emb_type == 'bert':
                        if 'test_data' not in emb_tune_params and test:
                            emb_tune_params['test_data'] = (test[0], test[-1])
                        emb_tune_params['save_to'] = emb_path if emb_path else \
                                                     emb_path_prefix
                        if emb_model_device \
                       and 'device' not in emb_tune_params:
                            emb_tune_params['device'] = emb_model_device
                        if 'seed' not in emb_tune_params:
                            emb_tune_params['seed'] = seed
                        if 'log_file' not in emb_tune_params:
                            emb_tune_params['log_file'] = log_file
                        if log_file:
                            print(file=log_file)
                        emb_path = WordEmbeddings.bert_tune(
                            train[0], train[-1], **emb_tune_params
                        )['model_name']
                    else:
                        raise ValueError(
                            "ERROR: tune method for '{}' embeddings "
                                .format(emb_type)
                          + 'is not implemented'
                       )
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
                        emb_params.get('emb_type',
                            emb_params['word_emb_type']),
                        emb_params.get('emb_path',
                            emb_params['word_emb_path']),
                        emb_model_device=
                            emb_params.get('emb_model_device',
                                emb_params.get('word_emb_model_device'),
                                    word_emb_model_device),
                        emb_tune_params=
                            emb_params.get('emb_tune_params',
                                emb_params.get('word_emb_tune_params'))
                    )

            if seed:
                junky.enforce_reproducibility(seed=seed)

            # 3. Create datasets
            if log_file:
                print('\nCREATE DATASETS', file=log_file)
            self._ds = self._create_dataset(
                train[0],
                word_emb_type=word_emb_type, word_emb_path=word_emb_path,
                word_emb_model_device=word_emb_model_device,
                word_transform_kwargs=word_transform_kwargs,
                word_next_emb_params=word_next_emb_params,
                with_chars=rnn_emb_dim or cnn_emb_dim, tags=train[1:-1],
                labels=train[-1])
            self._save_dataset(model_name)
            if test:
                ds_test = self._ds.clone(with_data=False)
                self._transform_dataset(test[0], tags=test[1:-1],
                                        labels=test[-1], ds=ds_test)
            else:
                ds_test = None

            # 4. Create model
            if log_file:
                print('\nCREATE A MODEL', file=log_file)
            self._model, criterion, optimizer, scheduler = \
                FeatsTaggerModel.create_model_for_train(
                    len(self._ds.get_dataset('y').transform_dict),
                    tags_pad_idx=self._ds.get_dataset('y').pad,
                    vec_emb_dim=self._ds.get_dataset('x').vec_size
                                    if word_emb_type is not None else
                                None,
                    alphabet_size=len(self._ds.get_dataset('x_ch')
                                          .transform_dict)
                                      if rnn_emb_dim or cnn_emb_dim else
                                  0,
                    char_pad_idx=self._ds.get_dataset('x_ch').pad
                                     if rnn_emb_dim or cnn_emb_dim else
                                 0,
                    rnn_emb_dim=rnn_emb_dim, cnn_emb_dim=cnn_emb_dim,
                    cnn_kernels=cnn_kernels, upos_emb_dim=upos_emb_dim,
                    num_upos=num_upos, upos_pad_idx=upos_pad_idx,
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
            criterion, optimizer, scheduler = \
                self._model.adjust_model_for_tune()
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
            total_res[feat] = res

            del self._model
            del self._ds

        return total_res
