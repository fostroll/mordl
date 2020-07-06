# -*- coding: utf-8 -*-
# MorDL project: Base tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import junky
from mordl.utils import LOG_FILE
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
        self._dataset = None

    @staticmethod
    def _get_filenames(model_name):
        if isinstance(model_name, tuple):
            model_fn, model_config_fn, dataset_fn, dataset_config_fn = \
                model_name
        else:
            if model_name.endswith('.pt'):
                model_name = model.name[:-3]
            model_fn, model_config_fn, dataset_fn, dataset_config_fn = \
                model_name + '.pt'   , model_name + '.config.json' \
                model_name + '_ds.pt', model_name + '_ds.config.json'
        return model_fn, model_config_fn, dataset_fn, dataset_config_fn

    @staticmethod
    def _prepare_corpus(corpus, tags_to_remove=None):
        return junky.extract_conllu_fields(
            junky.conllu_remove(corpus, remove=tags_to_remove),
            fields=['UPOS']
        )

    @staticmethod
    def _create_dataset(
        sentences, word_emb_type=None, word_emb_path=None,
        word_emb_model_device=None, word_transform_kwargs=None,
        word_next_emb_params=None, with_chars=False, labels=None
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

        if labels:
            y = TokenDataset(labels, pad_token='<PAD>', transform=True,
                             keep_empty=False)
            ds.add('y', y, with_lens=False)

        return ds

    @classmethod
    def _save_dataset(cls, ds, model_name):
        ds_fn, ds_config_fn = cls._get_filenames(model_name)[2:4]
        config = {}
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            cfg = getattr(ds_, CONFIG_ATTR, None)
            if cfg:
                config[name] = cfg
        with open(ds_config_fn, 'wt', encoding='utf-8') as f:
            print(json.dumps(config, sort_keys=True, indent=4), file=f)
        ds.save(ds_fn, with_data=with_data)

    @classmethod
    def _load_dataset(cls, model_name, device=None):
        ds_fn, ds_config_fn = cls._get_filenames(model_name)[2:4]
        ds = FrameDataset.load(ds_fn)
        with open(ds_config_file, 'rt', encoding='utf-8') as f:
            json.loads(f.read())
        for name, cfg in config.items():
            WordEmbeddings.apply_config(ds.get_dataset(name), cfg)
        if device:
            ds.to(device)
        return ds

    @staticmethod
    def _transform_dataset(ds, sentences, labels=None):
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            if name != 'y':
                if not WordEmbeddings.transform_dataset(ds_, sentences):
                    ds_.transform(sentences)
            elif labels:
                ds_.transform(labels)

    def save(self, model_name, log_file=LOG_FILE):
        self._save_dataset(self._dataset, model_name)
        model_fn, model_config_fn = cls._get_filenames(model_name)[:2]
        self._model.save_config(model_config_fn, log_file=log_file)
        self._model.save_state_dict(model_fn, log_file=log_file)

    def load(self, model_class, model_name, device=None, dataset_device=None,
             log_file=LOG_FILE):
        self._dataset = self._load_dataset(model_name, device=dataset_device)
        model_fn, model_config_fn = cls._get_filenames(model_name)[:2]
        self._model = model_class.create_from_config(
            model_config_fn, state_dict_f=model_fn, device=device,
            log_file=log_file
        )

    def load_train_corpus(self, corpus, append=False, test=None, seed=None):
        super().load_train_corpus(corpus, append=append, parse=False,
                                  test=test, seed=seed)
