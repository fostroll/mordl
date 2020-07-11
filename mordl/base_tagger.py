# -*- coding: utf-8 -*-
# MorDL project: Base tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from corpuscula.corpus_utils import _AbstractCorpus
from copy import deepcopy
import itertools
import json
import junky
from junky.dataset import CharDataset, DummyDataset, FrameDataset, \
                          LenDataset, TokenDataset
from mordl import WordEmbeddings
from morra.base_parser import BaseParser
from mordl.defaults import BATCH_SIZE, CONFIG_ATTR, LOG_FILE, TRAIN_BATCH_SIZE
import sys
import torch
from typing import Iterator


class BaseTagger(BaseParser):
    """"""
    def __err_hideattr(self, name):
        raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(self.__class__.__name__, name))
    def __err_override(self, name):
        raise AttributeError(("If you want to call the method '{}'() "
                              "of class '{}', you must override it")
                                  .format(name, self.__class__.__name__))
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
        self._model = None
        self._ds = None

    def load_train_corpus(self, corpus, append=False, test=None, seed=None):
        super().load_train_corpus(corpus, append=append, parse=False,
                                  test=test, seed=seed)

    @classmethod
    def _get_corpus(cls, corpus, asis=False, log_file=LOG_FILE):
        if isinstance(corpus, str):
            corpus = cls.load_conllu(corpus, log_file=log_file)
        elif (isinstance(corpus, type)
          and issubclass(corpus, _AbstractCorpus)) \
          or isinstance(corpus, _AbstractCorpus):
            corpus = corpus.test()
        elif callable(corpus):
            corpus = corpus()
        return (s[0] if not asis and isinstance(s, tuple) else s
                    for s in corpus)

    @staticmethod
    def _get_filenames(model_name):
        if isinstance(model_name, tuple):
            model_fn, model_config_fn, ds_fn, ds_config_fn = model_name
        else:
            if model_name.endswith('.pt'):
                model_name = model.name[:-3]
            model_fn, model_config_fn, ds_fn, ds_config_fn = \
                model_name + '.pt', model_name + '.config.json', \
                model_name + '_ds.pt', model_name + '_ds.config.json'
        return model_fn, model_config_fn, ds_fn, ds_config_fn

    @staticmethod
    def _prepare_corpus(corpus, fields, tags_to_remove=None):
        return junky.extract_conllu_fields(
            junky.conllu_remove(corpus, remove=tags_to_remove),
            fields=fields
        )

    @staticmethod
    def _create_dataset(
        sentences, word_emb_type=None, word_emb_path=None,
        word_emb_model_device=None, word_transform_kwargs=None,
        word_next_emb_params=None, with_chars=False, tags=None,
        labels=None, log_file=LOG_FILE
    ):
        ds = FrameDataset()

        if word_emb_type is not None:
            x = WordEmbeddings.create_dataset(
                sentences, emb_type=word_emb_type, emb_path=word_emb_path,
                emb_model_device=word_emb_model_device,
                transform_kwargs=word_transform_kwargs,
                next_emb_params=word_next_emb_params, log_file=log_file
            )
            ds.add('x', x)
        else:
            ds.add('x', DummyDataset(data=sentences))
            ds.add('x_lens', LenDataset(data=sentences))

        if with_chars:
            x_ch = CharDataset(sentences,
                               unk_token='<UNK>', pad_token='<PAD>',
                               transform=True)
            ds.add('x_ch', x_ch, with_lens=False)
        else:
            ds.add('x_ch', DummyDataset(data=sentences))
            ds.add('x_ch_lens', DummyDataset(data=sentences))

        if tags:
            for i, t_ in enumerate(tags):
                t = TokenDataset(t_, pad_token='<PAD>', transform=True,
                                 keep_empty=False)
                ds.add('t_{}'.format(i), t, with_lens=False)

        if labels:
            y = TokenDataset(labels, pad_token='<PAD>', transform=True,
                             keep_empty=False)
            ds.add('y', y, with_lens=False)

        return ds

    def _transform(self, sentences, tags=None, labels=None, ds=None,
                   batch_size=BATCH_SIZE, log_file=LOG_FILE):
        if ds is None:
            ds = self._ds
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                if not WordEmbeddings.transform(
                    ds_, sentences, batch_size=batch_size, log_file=log_file
                ):
                    ds_.transform(sentences)
            elif typ == 't':
                ds_.transform(tags[int(idx)])
            elif labels and typ == 'y':
                ds_.transform(labels)

    def _transform_collate(self, sentences, tags=None, labels=None,
                           batch_size=BATCH_SIZE, log_file=LOG_FILE):
        res = []
        for name in self._ds.list():
            ds_ = self._ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                res_ = WordEmbeddings.transform_collate(
                    ds_, sentences, batch_size=batch_size, log_file=log_file
                )
                if not res_:
                    res_ = ds_.transform_collate(sentences,
                                                 batch_size=batch_size)
                res.append(res_)
            elif typ == 't':
                res.append(ds_.transform_collate(
                    tags[int(idx)], batch_size=batch_size,
                    collate_kwargs={'with_lens': False}
                ))
            elif labels and typ == 'y':
                res.append(ds_.transform_collate(
                    labels, batch_size=batch_size,
                    collate_kwargs={'with_lens': False}
                ))
        for batch in zip(*res):
            batch_ = []
            for res_ in batch:
                batch_.extend(res_) if isinstance(res_, tuple) else \
                batch_.append(res_)
            yield batch_

    def _save_dataset(self, model_name):
        ds_fn, ds_config_fn = self._get_filenames(model_name)[2:4]
        config = {}
        for name in self._ds.list():
            ds_ = self._ds.get_dataset(name)
            cfg = getattr(ds_, CONFIG_ATTR, None)
            if cfg:
                config[name] = cfg
        with open(ds_config_fn, 'wt', encoding='utf-8') as f:
            print(json.dumps(config, sort_keys=True, indent=4), file=f)
        self._ds.save(ds_fn, with_data=False)

    def _load_dataset(self, model_name, device=None, log_file=LOG_FILE):
        ds_fn, ds_config_fn = self._get_filenames(model_name)[2:4]
        if log_file:
            print('Loading dataset...', end=' ', file=log_file)
            log_file.flush()
        self._ds = FrameDataset.load(ds_fn)
        with open(ds_config_fn, 'rt', encoding='utf-8') as f:
            config = json.loads(f.read())
            for name, cfg in config.items():
                WordEmbeddings.apply_config(self._ds.get_dataset(name), cfg)
            if device:
                self._ds.to(device)
        if log_file:
            print('done.', file=log_file)

    def save(self, model_name, log_file=LOG_FILE):
        assert self._ds, "ERROR: the tagger doesn't have a dataset to save"
        assert self._model, "ERROR: the tagger doesn't have a model to save"
        self._save_dataset(model_name)
        model_fn, model_config_fn = cls._get_filenames(model_name)[:2]
        self._model.save_config(model_config_fn, log_file=log_file)
        self._model.save_state_dict(model_fn, log_file=log_file)

    def load(self, model_class, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
        self._load_dataset(model_name, device=dataset_device,
                           log_file=log_file)
        model_fn, model_config_fn = self._get_filenames(model_name)[:2]
        self._model = model_class.create_from_config(
            model_config_fn, state_dict_f=model_fn, device=device,
            log_file=log_file
        )

    def predict(self, field, add_fields, corpus, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE):
        assert self._ds is not None, \
               "ERROR: the tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert self._model, \
               "ERROR: the tagger doesn't have a model. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'

        if field.count(':') == 1:
            field += ':_'

        def process(corpus):
            corpus = self._get_corpus(corpus, asis=True, log_file=log_file)
            device = next(self._model.parameters()).device or junky.CPU

            ds_y = self._ds.get_dataset('y')
            if clone_ds:
                ds = self._ds.clone()
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
                        corpus_, fields=add_fields,
                        with_empty=True, return_nones=True
                    )
                sentences, tags, empties, nones = \
                    res[0], res[1:-2], res[-2], res[-1]
                if clone_ds:
                    self._transform(
                        sentences, tags=tags, batch_size=batch_size, ds=ds,
                        log_file=log_file
                    )
                    loader = ds.create_loader(batch_size=batch_size,
                                              shuffle=False)
                else:
                    loader = self._transform_collate(
                        sentences, tags=tags, batch_size=batch_size,
                        log_file=log_file
                    )
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
                            res_corpus_, field, values,
                            empties=empties, nones=nones
                        )
                    ):
                        yield sentence, orig_sentence
                else:
                    for sentence in junky.embed_conllu_fields(
                        corpus_, field, values,
                        empties=empties, nones=nones
                    ):
                        yield sentence

        corpus = process(corpus)
        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, field, gold, test=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):

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
            print('Evaluate ' + header, file=LOG_FILE)
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
                print('Nothing to do!', file=LOG_FILE)
            else:
                sp = ' ' * (len(header) - 2)
                print(header + ' total: {}'.format(nt), file=LOG_FILE)
                print(sp   + ' correct: {}'.format(ct), file=LOG_FILE)
                print(sp   + '   wrong: {}{}'.format(
                    nt - ct, ' [{} excess / {} absent{}]'.format(
                        ce, ca, '' if label else ' / {} wrong type'.format(cr)
                    ) if nt != n else ''
                ), file=LOG_FILE)
                print(sp   + 'Accuracy: {}'.format(ct / nt if nt > 0 else 1.))
                if nt != n:
                    print('[Total accuracy: {}]'
                              .format(c / n if n > 0 else 1.))
        return ct / nt if nt > 0 else 1.

    def train(self, field, add_fields, model_class, tag_emb_names, model_name,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              seed=None, log_file=LOG_FILE, **model_kwargs):

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if isinstance(tag_emb_names, str):
            tag_emb_names = [tag_emb_names]

        if field.count(':') == 1:
            field += ':_'
        header = ':'.join(field.split(':')[:2])
        bert_header = header.lower().replace(':', '-') + '_'
        fields = [] if add_fields is None else \
                       [add_fields] if isinstance(add_fields, str) else \
                       add_fields
        fields.append(field)

        if log_file:
            print('=== {} TAGGER TRAINING PIPELINE ==='.format(header))

        model_fn, model_config_fn = self._get_filenames(model_name)[:2]

        # 1. Prepare corpora
        train = self._prepare_corpus(
            self._train_corpus, fields=fields,
            tags_to_remove=tags_to_remove
        )
        test = self._prepare_corpus(
            self._test_corpus, fields=fields,
            tags_to_remove=tags_to_remove
        ) if self._test_corpus is not None else [None]

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
                        emb_tune_params['test_data'] = (test[0], test[-1])
                    emb_tune_params['save_to'] = emb_path if emb_path else \
                                                 bert_header
                    if emb_model_device and 'device' not in emb_tune_params:
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
            print('\nDATASETS CREATION', file=log_file)
        self._ds = self._create_dataset(
            train[0],
            word_emb_type=word_emb_type, word_emb_path=word_emb_path,
            word_emb_model_device=word_emb_model_device,
            word_transform_kwargs=word_transform_kwargs,
            word_next_emb_params=word_next_emb_params,
            with_chars=model_kwargs.get('rnn_emb_dim') \
                    or model_kwargs.get('cnn_emb_dim'),
            tags=train[1:-1], labels=train[-1], log_file=sys.stderr)
        self._save_dataset(model_name)
        if test:
            ds_test = self._ds.clone(with_data=False)
            self._transform(test[0], tags=test[1:-1], labels=test[-1],
                            ds=ds_test)
        else:
            ds_test = None

        # 4. Create model
        if log_file:
            print('\nMODEL CREATION', file=log_file)
        ds_ = self._ds.get_dataset('y')
        model_args = [len(ds_.transform_dict)]
        model_kwargs['labels_pad_idx'] = ds_.pad
        if word_emb_type:
            ds_ = self._ds.get_dataset('x')
            model_kwargs['vec_emb_dim'] = ds_.vec_size
        if model_kwargs.get('rnn_emb_dim') or model_kwargs.get('cnn_emb_dim'):
            ds_ = self._ds.get_dataset('x_ch')
            model_kwargs['alphabet_size'] = len(ds_.transform_dict)
            model_kwargs['char_pad_idx'] = ds_.pad
        if tag_emb_names:
            ds_list = self._ds.list()
            names = iter(tag_emb_names)
            for name in self._ds.list():
                if name.startswith('t_'):
                    ds_ = self._ds.get_dataset('name')
                    name = next(tag_emb_names)
                    emb_dim = model_kwargs[name + '_emb_dim']
                    if emb_dim:
                        model_kwargs[name + '_num'] = len(ds_.transform_dict)
                        model_kwargs[name + '_pad_idx'] = ds_.pad
        self._model, criterion, optimizer, scheduler = \
            model_class.create_model_for_train(*model_args, **model_kwargs)
        if device:
            self._model = self._model.to(device)
        self._model.save_config(model_config_fn, log_file=log_file)

        # 5. Train model
        if log_file:
            print('\nMODEL TRAINING', file=log_file)
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
            print('\nMODEL TUNING', file=log_file)
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
