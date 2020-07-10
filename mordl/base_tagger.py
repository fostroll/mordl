# -*- coding: utf-8 -*-
# MorDL project: Base tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from corpuscula.corpus_utils import _AbstractCorpus
import json
import junky
from junky.dataset import CharDataset, DummyDataset, FrameDataset, \
                          LenDataset, TokenDataset
from mordl import WordEmbeddings
from mordl.utils import CONFIG_ATTR, LOG_FILE
from morra.base_parser import BaseParser


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

    def _transform_dataset(self, sentences, tags=None, labels=None, ds=None):
        if ds is None:
            ds = self._ds
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                if not WordEmbeddings.transform_dataset(ds_, sentences):
                    ds_.transform(sentences)
            elif typ == 't':
                ds_.transform(tags[int(idx)])
            elif labels and typ == 'y':
                ds_.transform(labels)

    def _transform_collate_dataset(self, sentences, tags=None, labels=None,
                                   batch_size=64):
        res = []
        for name in self._ds.list():
            ds_ = self._ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                res_ = WordEmbeddings.transform_collate_dataset(
                    ds_, sentences, batch_size=batch_size
                )
                if not res_:
                    res_ = ds_.transform_collate(sentences,
                                                 batch_size=batch_size)
                res.extend(res_) if isinstance(res_, tuple) else \
                res.append(res_)
            elif typ == 't':
                res_ = ds_.transform_collate(tags[int(idx)],
                                             batch_size=batch_size)
                res.extend(res_) if isinstance(res_, tuple) else \
                res.append(res_)
            elif labels and typ == 'y':
                res_ = ds_.transform_collate(labels, batch_size=batch_size)
                res.extend(res_) if isinstance(res_, tuple) else \
                res.append(res_)
        return zip(*res)

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

    def _load_dataset(self, model_name, device=None):
        ds_fn, ds_config_fn = self._get_filenames(model_name)[2:4]
        self._ds = FrameDataset.load(ds_fn)
        with open(ds_config_fn, 'rt', encoding='utf-8') as f:
            config = json.loads(f.read())
            for name, cfg in config.items():
                WordEmbeddings.apply_config(self._ds.get_dataset(name), cfg)
            if device:
                self._ds.to(device)

    def save(self, model_name, log_file=LOG_FILE):
        assert self._ds, "ERROR: the tagger doesn't have a dataset to save"
        assert self._model, "ERROR: the tagger doesn't have a model to save"
        self._save_dataset(model_name)
        model_fn, model_config_fn = cls._get_filenames(model_name)[:2]
        self._model.save_config(model_config_fn, log_file=log_file)
        self._model.save_state_dict(model_fn, log_file=log_file)

    def load(self, model_class, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
        self._load_dataset(model_name, device=dataset_device)
        model_fn, model_config_fn = self._get_filenames(model_name)[:2]
        self._model = model_class.create_from_config(
            model_config_fn, state_dict_f=model_fn, device=device,
            log_file=log_file
        )
