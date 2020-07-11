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
from mordl.utils import CONFIG_ATTR, LOG_FILE
from morra.base_parser import BaseParser
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
        labels=None
    ):
        ds = FrameDataset()

        if word_emb_type is not None:
            x = WordEmbeddings.create_dataset(
                sentences, emb_type=word_emb_type, emb_path=word_emb_path,
                emb_model_device=word_emb_model_device,
                transform_kwargs=word_transform_kwargs,
                next_emb_params=word_next_emb_params
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
                   log_file=LOG_FILE):
        if ds is None:
            ds = self._ds
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                if not WordEmbeddings.transform(
                    ds_, sentences,
                    transform_kwargs={} if log_file else {'loglevel': 0}
                ):
                    ds_.transform(sentences)
            elif typ == 't':
                ds_.transform(tags[int(idx)])
            elif labels and typ == 'y':
                ds_.transform(labels)

    def _transform_collate(self, sentences, tags=None, labels=None,
                           batch_size=64, log_file=LOG_FILE):
        res = []
        for name in self._ds.list():
            ds_ = self._ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                res_ = WordEmbeddings.transform_collate(
                    ds_, sentences, batch_size=batch_size,
                    transform_kwargs={} if log_file else {'loglevel': 0}
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
                batch_size=64, split=None, clone_ds=False, save_to=None,
                log_file=LOG_FILE):
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
                    self._transform(sentences, tags=tags, ds=ds,
                                    log_file=log_file)
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

    def evaluate(self, field, add_fields, gold, test=None, label=None,
                 batch_size=64, split=None, clone_ds=False,
                 log_file=LOG_FILE):

        gold = self._get_corpus(gold, log_file=log_file)
        corpora = zip(gold, self._get_corpus(test, log_file=log_file)) \
                      if test else \
                  self.predict(gold, with_orig=True,
                               batch_size=batch_size, split=split,
                               clone_ds=clone_ds, log_file=log_file)
        field = field.split(':')
        val = field[1] if len(field) > 1 else None
        field = field[0]
        header = field
        if val:
            header += ':' + val
        if log_file:
            print('Evaluate ' + header, file=LOG_FILE)
        n = c = nt = ct = ca = ce = cr = 0
        i = -1
        for i, sentences in enumerate(corpora):
            for gold_token, test_token in zip(*sentences):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:


                    gold_val = gold_token[field]
                    test_val = test_token[field]
                    if val:
                        gold_val = gold_val.get(val)
                        test_val = test_val.get(val)
                    n += 1
                    if (label and (gold_val == label or test_val == label)) \
                    or (not label and (gold_val or test_val)):
                        nt += 1
                        if gold_val == test_val:
                            c += 1
                            ct += 1
                        elif not gold_val or (label and gold_val != label):
                            ce += 1
                        elif not test_val or (label and test_val != label):
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
