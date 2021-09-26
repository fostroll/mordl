# -*- coding: utf-8 -*-
# MorDL project: Word embeddings helper
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a class for loading and applying pretrained embeddings to
input sentences and creating datasets and collated dataloaders.
"""
import logging
import sys
#logging.basicConfig(level=logging.ERROR)
#if not sys.warnoptions:
#    import warnings
#    warnings.simplefilter('ignore')
#    os.environ['PYTHONWARNINGS'] = 'ignore'
from collections.abc import Iterable
from copy import deepcopy
from corpuscula.corpus_utils import get_root_dir
#from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import json
import junky
from junky.dataset import BaseDataset, BertDataset, WordCatDataset, \
                          WordDataset
from junky.trainer import Trainer, TrainerConfig
from mordl2.defaults import BATCH_SIZE, CONFIG_ATTR, CONFIG_EXT, LOG_FILE
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            f1_score, precision_score, recall_score
from tempfile import mkstemp
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, AutoConfig, AutoModel, AutoTokenizer, \
                         get_linear_schedule_with_warmup
#BertConfig, BertForSequenceClassification, \
#                         BertForTokenClassification, BertTokenizer, \

_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS = junky.kwargs(
    max_len=0, batch_size=BATCH_SIZE, hidden_ids=-2,
    aggregate_hiddens_op='cat', aggregate_subtokens_op='absmax',
    to=junky.CPU, loglevel=1
)
_DEFAULT_DATASET_TRANSFORM_KWARGS = junky.kwargs(
    check_lower=True
)
_MAX_BAD_EPOCHS = 0

# to suppress transformers' warnings
#logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
#logging.getLogger('pytorch_pretrained_bert.tokenization').setLevel(logging.ERROR)


class WordEmbeddings:
    """
    The class that handles loading various pretrained word embeddings.
    """

    @staticmethod
    def _full_tune(model, model_save_as, datasets, sents_data,
                   save_as=None, epochs=3, batch_size=8,
                   control_metric='accuracy', transform_kwargs=None,
                   seed=None, log_file=LOG_FILE):
        """Method for finetuning base BERT model on custom data.

        Args:

        **train_sentences**: sequence of already tokenized sentences (of the
        `list([str])` format) that will be used to train the model.

        **train_labels**: list of labels (of `str`) that will be used to
        train the model.

        **test_data** (`tuple(<sentences>, <labels>)`): development test set to
        validate the model during training.

        **model_name** (`str`): pre-trained BERT model name or path to the
        model. Default `model_name='bert-base-multilingual-cased'`.

        **device**: device for the BERT model. Default device is CPU.

        **save_to** (`str`): a path where the finetuned BERT model will be
        saved. If not specified, it will be generated based on **model_name**
        and the model tune parameters. If **save_to** is ended on the `'_'`
        symbol, it will be treated as a prefix for the autogenerated name.

        **max_len** (`int`): maximum input sequence length. By default,
        restricted by BERT's positional embeddings, `max_len=512`. Don't make
        in higher.

        **epochs** (`int`): number of finetuning epochs. Default `epochs=3`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=8`.

        **special_tokens** (`str`|`list(<str>)`): additional special tokens
        for BERT tokenizer.

        **control_metric** (`str`): metric to control training. Allowed values
        are: 'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **seed** (`int`): random seed.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns the finetune statistics.
        """
        assert control_metric in ['accuracy', 'f1', 'loss'], \
           f'ERROR: Unknown control_metric "{control_metric}" ' \
            "(only 'accuracy', 'f1' and 'loss' are available)."
        assert save_as, 'ERROR: Undefined `save_as` param.'

        train_ds, test_ds = datasets if isinstance(datasets, tuple) else \
                            (datasets, None)
        train_sents, test_sents = \
            sents_data if isinstance(sents_data, tuple) else sents_data, None
        print(train_ds)
        print(test_ds)
        print(len(train_ds), len(train_sents))
        print(len(test_ds), len(test_sents))
        assert (test_ds and test_sents) or not (test_ds or test_sents), \
            'ERROR: given {} without {}.' \
                .format(*(('test dataset', 'test sentences') if test_ds else
                          ('test sentences', 'test dataset')))
        assert len(train_ds) == len(train_sents), \
            'ERROR: train dataset and train corpus have different sizes.'
        assert not test_ds or len(test_ds) == len(test_sents), \
            'ERROR: test dataset and test corpus have different sizes.'

        if log_file:
            print('BERT MODEL TUNING', file=log_file)

        kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
        if transform_kwargs:
            kwargs.update(transform_kwargs)
        transform_kwargs = kwargs

        if seed:
            junky.enforce_reproducibility(seed)

        class WordEmbeddingsModel(nn.Module):

            def __init__(self, bert_ds, model_head):
                super().__init__()

                self.ds = bert_ds
                self.emb_model = self_ds.model
                self.tokenizer = bert_ds.tokenizer
                self.model_head = model_head

            def save_pretrained(self, paths):
                self.emb_model.save_pretrained(paths[0])
                self.model_head.save_state_dict(paths[1])

            def forward(self, sentences, *args, labels=None):
                x = self.ds.transform(
                    sentences, max_len=transform_kwargs.max_len,
                    batch_size=batch_size,
                    hidden_ids=transform_kwargs.hidden_ids,
                    aggregate_hiddens_op=\
                        transform_kwargs.aggregate_hiddens_op,
                    aggregate_subtokens_op=\
                        transform_kwargs.aggregate_subtokens_op,
                    with_grad=True, save=False, loglevel=0
                )
                x, = self.ds._collate(x, with_lens=False)
                return self.model_head(x, *args, labels=labels)

        train_ds_x = BaseDataset(train_sents)
        test_ds_x = BaseDataset(test_sents)

        train_ds_bert = train_ds.datasets['x']
        test_ds_bert = test_ds.datasets['x']
        bert_ds = train_ds_bert[0] = train_ds_bert[0].clone(with_data=False)
        test_ds_bert[0] = bert_ds.clone()

        train_ds.datasets['x'] = (train_ds_x, 1, {})
        test_ds.datasets['x'] = (test_ds_x, 1, {})

        train_dl = train_ds.create_dataloader(
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        test_dl = train_ds.create_dataloader(
            batch_size=transform_kwargs.batch_size, shuffle=False,
            num_workers=0
        )

        full_model = WordEmbeddingsModel(bert_ds, model)

        '''TODO: after training recreate datasets
        emb_config = AutoConfig.from_pretrained(
            bert_save_to, output_hidden_states=True,
            output_attentions=False
        )
        emb_model = AutoModel.from_pretrained(bert_save_to, config=config)
        train_ds_bert[0].model = test_ds_bert[0].model = emb_model
        train_ds_bert[0].transform(train_sents, **transform_kwargs)
        test_ds_bert[0].transform(test_sents, **transform_kwargs)
        '''

        if log_file:
            print("Loading model '{}'...".format(model_name), end=' ',
                  file=log_file)
            log_file.flush()
        if device:
            model.to(device)
        if log_file:
            print('done.', file=log_file)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                                  if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .01},
                {'params': [p for n, p in param_optimizer
                                  if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = \
                [{'params': [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5,
                          betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                          amsgrad=False)

        grad_norm_clip = 1.

        # Total number of training steps is
        # number of batches * number of epochs
        total_steps = len(train_loader) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
        # scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        trainer_config = TrainerConfig(
            (save_as, model_save_as), max_epochs=epochs,
            batch_lens_idx=1, batch_labels_idx=2,
            model_args=[0, 1], model_kwargs={'labels': -2},
            output_logits_idx=0, output_loss_idx=1,
            grad_norm_clip=grad_norm_clip
        )
        trainer = Trainer(
            trainer_config, full_model, train_dl, test_dataloader=test_dl,
            optimizer=optimizer, scheduler=scheduler,
            save_ckpt_method = \
                lambda model, paths: full_model.save_pretrained(paths),
            postprocess_method = 'strip_mask',
        #     force_cpu=True
        )

        try:
            res = trainer.train()
        except RuntimeError as e:
            if e.args and e.args[0].startswith('CUDA out of memory'):
                e = RuntimeError(
                    e.args[0] + '. To avoid this, consider to decrease '
                                'batch_size or max_len value(s)',
                    *e.args[1:]
                )
            raise e

        return res

    @staticmethod
    def load(emb_type, emb_path, emb_model_device=None, embs=None):
        """Method to load pretrained embeddings model from **emb_path**.

        Args:

        **emb_type**: (`str`) one of the supported embeddings types. Allowed
        values: 'bert' for *BERT*, 'ft' for *FastText*, 'glove' for *Glove*,
        'w2v' for *Word2vec*.

        **emb_path**: path to the embeddings file.

        **emb_model_device**: relevant with `emb_type='bert'`. The device
        where the BERT model will be loaded to.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If **emb_path** is in
        **embs**, the method just return the corresponding model.

        Returns the embeddings model.
        """
        if embs and emb_path in embs:
            model = embs[emb_path]
        else:
            emb_type = emb_type.lower()
            if emb_type == 'bert':
                tokenizer = AutoTokenizer.from_pretrained(
                    emb_path, do_lower_case=False
                )
                config = AutoConfig.from_pretrained(
                    emb_path, output_hidden_states=True,
                    output_attentions=False
                )
                model = AutoModel.from_pretrained(emb_path, config=config)
                if emb_model_device:
                    model.to(emb_model_device)
                model.eval()
                model = model, tokenizer

            elif emb_type == 'glove':
                try:
                    model = KeyedVectors.load_word2vec_format(emb_path,
                                                              binary=False)
                except ValueError:
                    fn = os.path.basename(emb_path)
                    pref, suff = os.path.splitext(fn)
                    dn = get_root_dir()
                    f, fn = mkstemp(suffix=suff, prefix=pref, dir=nd)
                    f.close()
                    glove2word2vec(emb_path, fn)
                    model = KeyedVectors.load_word2vec_format(fn,
                                                              binary=False)
                    os.remove(fn)
                except UnicodeDecodeError:
                    model = KeyedVectors.load(emb_path)
                model = {x: model.vectors[y.index]
                             for x, y in model.vocab.items()}

            elif emb_type in ['ft', 'fasttext']:
                try:
                    #model = load_facebook_model(emb_path).wv
                    model = load_facebook_vectors(emb_path)
                except NotImplementedError:
                    model = KeyedVectors.load(emb_path)
                    if not isinstance(model, FastTextKeyedVectors):
                        raise ValueError('ERROR: Unable to load '
                                         'Word2vec vectors as FastText')

            elif emb_type in ['w2v', 'word2vec']:
                try:
                    model = KeyedVectors.load_word2vec_format(emb_path,
                                                              binary=False)
                except UnicodeDecodeError:
                    try:
                        model = KeyedVectors.load_word2vec_format(emb_path,
                                                                  binary=True)
                    except UnicodeDecodeError:
                        try:
                            model = load_facebook_model(emb_path).wv
                        except NotImplementedError:
                            model = KeyedVectors.load(emb_path)
                model = {x: model.vectors[y.index]
                             for x, y in model.vocab.items()}

            else:
                raise ValueError('ERROR: Unknown emb_type. Allowed values: '
                                 "'bert' for BERT, 'glove' for GloVe, "
                                 "'ft' for fastText, 'w2v' for Word2vec")

            if embs is not None:
                embs[emb_path] = model
        return model

    @classmethod
    def create_dataset(cls, sentences, emb_type='ft', emb_path=None,
                       emb_model_device=None, transform_kwargs=None,
                       next_emb_params=None, embs=None, loglevel=2):
        """Creates dataset with embedded sequences.

        Args:

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format) that will be used for initial transform of
        the created dataset.

        **emb_type**: one of ('bert'|'glove'|'ft'|'w2v') embedding types.

        **emb_path** (`str`): path to word embeddings storage.

        **emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert').

        **transform_kwargs** (`dict`): keyword arguments for `.transform()`
        method of the dataset created for sentences to word embeddings
        conversion. See the `.transform()` method of
        `junky.datasets.BertDataset` for the the description of the
        parameters.

        **next_emb_params**: if you want to use several different embedding
        models at once, pass parameters of the additional model as a
        dictionary with keys `(emb_path, emb_model_device, transform_kwargs)`;
        or a list of such dictionaries if you need more than one additional
        models.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If **emb_path** is in
        **embs**, the method don't load the corresponding model and use
        already loaded one.

        **loglevel**: param for dataset's `.transform()`. Relevant with
        `emb_type='bert'`.

        Returns a loaded dataset.
        """
        emb_params = \
            [(emb_type, emb_path, emb_model_device, transform_kwargs)]
        if next_emb_params:
            if isinstance(next_emb_params, dict) \
            or isinstance(next_emb_params[0], str):
                next_emb_params = [next_emb_params]
            for emb_params_ in next_emb_params:
                if isinstance(emb_params_, dict):
                    emb_params.append(
                        (emb_params_.get('emb_type',
                         emb_params_.get('word_emb_type',
                         emb_params_['emb_type'])),  # for correct value name
                                                     # in error message
                         emb_params_.get('emb_path',
                         emb_params_.get('word_emb_path',
                         emb_params_['emb_path'])),  # for correct value name
                                                     # in error message
                         emb_params_.get('emb_model_device',
                         emb_params_.get('word_emb_model_device')),

                         emb_params_.get('transform_kwargs',
                         emb_params_.get('word_transform_kwargs')))
                     )
                else:
                    emb_params.append(emb_params_)

        ds, config = [], []
        for emb_params_ in emb_params:
            emb_type, emb_path = emb_params_[:2]
            emb_model_device = emb_params_[2] if len(emb_params_) > 2 else \
                               None
            transform_kwargs = emb_params_[3] if len(emb_params_) > 3 else \
                               None
            if isinstance(emb_model_device, dict) or (
                transform_kwargs and not isinstance(transform_kwargs, dict)
            ):
                emb_model_device, transform_kwargs = \
                    transform_kwargs, emb_model_device

            config.append(junky.kwargs_nonempty(
                emb_type=emb_type, emb_path=emb_path,
                emb_model_device=str(emb_model_device),
                transform_kwargs=transform_kwargs
            ))

            if transform_kwargs is None:
                transform_kwargs = {}

            emb_model = cls.load(emb_type, emb_path,
                                 emb_model_device=emb_model_device, embs=embs)

            if emb_type == 'bert':
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                if loglevel is not None:
                    kwargs['loglevel'] = loglevel
                transform_kwargs = kwargs
                model, tokenizer = emb_model
                x = BertDataset(model, tokenizer)
            else:
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
                x = WordDataset(
                    emb_model=emb_model,
                    vec_size=len(next(iter(emb_model.values())))
                                 if isinstance(emb_model, dict) else
                             emb_model.vector_size,
                    unk_token='<UNK>', pad_token='<PAD>'
                )
            junky.clear_tqdm()
            x.transform(sentences, **transform_kwargs)
            ds.append(x)

        if len(ds) == 1:
            ds, config = ds[0], config[0]
        else:
            ds_ = WordCatDataset()
            for i, x in enumerate(ds):
                ds_.add('x_{}'.format(i), x)
            ds = ds_

        if hasattr(ds, CONFIG_ATTR):
            raise AttributeError(
                'ERROR: {} class has unexpected attribute {}'
                    .format(ds.__class__, CONFIG_ATTR)
            )
        setattr(ds, CONFIG_ATTR, config)

        return ds

    @classmethod
    def transform(cls, ds, sentences, transform_kwargs=None,
                  batch_size=BATCH_SIZE, loglevel=1):
        """Converts **sentences** of tokens to the sequences of the
        corresponding indices.

        Args:

        **ds**: a dataset to transform **sentences**.

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format).

        **transform_kwargs**: keyword arguments for `.transform` function.

        **batch_size**: number of sentences per batch. If not `None`,
        overwrites `batch_size` parameter from `transform_kwargs`.
        By default, `batch size=64`.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns `True` if **ds** is of appropriate type and the transforming
        have been done). Otherwise, returns `False`, that means that you have
        to transform data by it's own method.
        """
        res = False
        config = getattr(ds, CONFIG_ATTR, None)
        if config:
            if isinstance(ds, WordCatDataset):
                for name, cfg in zip(ds.list(), config):
                    res = res \
                      and cls.transform_dataset(ds.get_dataset(name),
                                                sentences)
                    if not res:
                        break
            else:
                kwargs = config.get('transform_kwargs', {})
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
        for _ in range(1):
            if isinstance(ds, BertDataset):
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if batch_size:
                    kwargs['batch_size'] = batch_size
                if loglevel is not None:
                    kwargs['loglevel'] = loglevel
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            elif isinstance(ds, WordDataset):
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            else:
                break
            junky.clear_tqdm()
            ds.transform(sentences, **transform_kwargs)
            res = True
        return res

    @classmethod
    def transform_collate(cls, ds, sentences, transform_kwargs=None,
                          batch_size=BATCH_SIZE, loglevel=1):
        """Sequentially makes batches from `sentences`.

        Args:

        **ds**: a dataset to transform **sentences**.

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format).

        **transform_kwargs**: keyword arguments for the `.transform` function
        of the **ds**.

        **batch_size**: number of sentences per batch. If not `None`,
        overwrites `batch_size` parameter from `transform_kwargs`.
        By default, batch size is set in defaults to `64`.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns a sequence of batches for training.
        """
        res = False
        config = getattr(ds, CONFIG_ATTR, None)
        if config:
            if isinstance(ds, WordCatDataset):
                for name, cfg in zip(ds.list(), config):
                    res = res \
                      and cls.transform_dataset(ds.get_dataset(name),
                                                sentences)
                    if not res:
                        break
            else:
                kwargs = config.get('transform_kwargs', {})
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
        for _ in range(1):
            if isinstance(ds, BertDataset):
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            elif isinstance(ds, WordDataset):
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    loglevel = transform_kwargs.pop('loglevel', loglevel)
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            else:
                break
            junky.clear_tqdm()
            res = ds.transform_collate(sentences, batch_size=batch_size,
                                       transform_kwargs=transform_kwargs,
                                       loglevel=loglevel)
        return res

    @staticmethod
    def save_dataset(ds, f, config_f=True):
        """Saves dataset to the specified file.

        Args:

        **ds**: a dataset to save.

        **f** (`str` : `file`): a file where the dataset will be saved.

        **config_f** (`str` : `file`): json config file.
        """
        if config_f is True and isinstance(f, str):
            pref, suff = os.path.splitext(f)
            config_f = pref + CONFIG_EXT
        if config_f:
            need_close = False
            if isinstance(config_f, str):
                config_f = open(config_f, 'wt', encoding='utf-8')
                need_close = True
            try:
                print(json.dumps(getattr(ds, CONFIG_ATTR, ()),
                                 sort_keys=True, indent=4),
                      file=config_f)
            finally:
                if need_close:
                    config_f.close()
        ds.save(f, with_data=False)

    @classmethod
    def load_dataset(cls, f, config_f=True, emb_path=None, device=None,
                     embs=None):
        """Loads previously saved dataset with embedded tokens.

        Args:

        **f** (`str` : `file`): a file where the dataset will be loaded from.

        **config_f** (`str` : `file`): json config file.

        **emb_path**: a path where dataset to load from if you want to
        override the value from config.

        **device**: a device for the loaded dataset if you want to override
        the value from config.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If value of `emb_path` from
        **config_f** is in **embs**, the method don't load the corresponding
        model and use already loaded one.

        Returns a loaded dataset.
        """
        #ds = BaseDataset.load()
        ds = WordDataset.load(f)  # sic!

        if config_f is True:
            assert isinstance(f, str), \
                   'ERROR: config_f can be True only with f of str type'
            pref, suff = os.path.splitext(f)
            config_f_ = pref + CONFIG_EXT
            if os.path.isfile(config_f_):
                config_f = config_f_
        if config_f:
            need_close = False
            if isinstance(config_f, str):
                config_f = open(config_f, 'wt', encoding='utf-8')
                need_close = True
            try:
                json.loads(config_f.read())
            finally:
                if need_close:
                    config_f.close()
        else:
            config = getattr(ds, CONFIG_ATTR, ())

        if emb_path:
            config['emb_path'] = emb_path
        if device:
            config['emb_model_device'] = device

        cls.apply_config(ds, config, embs=embs)
        return ds

    @classmethod
    def apply_config(cls, ds, config, emb_path=None, device=None, embs=None):
        """Apply config file to the dataset.

        Args:

        **ds**: a dataset to apply **config**.

        **config** (`dict` | `list([dict])`): config with model parameters.

        **emb_path**: a path where dataset to load from if you want to
        override the value from config.

        **device**: a device for the loaded dataset if you want to override
        the value from config.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If value of `emb_path` from
        **config** is in **embs**, the method don't load the corresponding
        model and use already loaded one.
        """
        if isinstance(config, dict):
            config = [config]

        assert len(config) == 1 or len(config) == len(ds.list()), \
               'ERROR: f and config_f have incompatible data'

        embs, xtrn = {} if embs is None else embs, []
        for cfg in config:
            emb_type = cfg['emb_type']
            if emb_path is None:
                emb_path = cfg['emb_path']
            emb_model_device = device if device else \
                               cfg.get('emb_model_device')
            transform_kwargs = cfg.get('transform_kwargs', {})
            model = cls.load(emb_type, emb_path,
                             emb_model_device=emb_model_device,
                             embs=embs)
            xtrn.append(model)

        if len(xtrn) == 1:
            xtrn = xtrn[0]
        else:
            xtrn = {x: y for x, y in zip(ds.list(), xtrn)}
        ds._push_xtrn(xtrn)

        return embs
