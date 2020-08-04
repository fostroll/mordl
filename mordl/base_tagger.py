# -*- coding: utf-8 -*-
# MorDL project: Base tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a base class for specialized morphological taggers.
"""
from corpuscula.corpus_utils import _AbstractCorpus
from copy import deepcopy
import itertools
import json
import junky
from junky.dataset import CharDataset, DummyDataset, FrameDataset, \
                          LabelDataset, LenDataset, TokenDataset
from mordl import WordEmbeddings
from morra.base_parser import BaseParser
from mordl.defaults import BATCH_SIZE, CONFIG_ATTR, CONFIG_EXT, LOG_FILE, \
                           NONE_TAG, TRAIN_BATCH_SIZE
from mordl.lib.conll18_ud_eval import main as _conll18_ud_eval
import os
import sys
import torch
from typing import Iterator

_MODEL_CONFIG_FN = 'model_config.json'
_MODEL_FN = 'model.pt'
_DATASETS_CONFIG_FN = 'ds_config.json'
_DATASETS_FN = 'ds.pt'
_CDICT_FN = 'cdict.pickle'


class BaseTagger(BaseParser):
    """
    A base class for the project's taggers.
    """

    def __err_hideattr(self, name):
        raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(self.__class__.__name__, name))
    def __err_override(self, name):
        raise AttributeError(("If you want to call the method '{}'() "
                              "of class '{}', you must override it")
                                  .format(name, self.__class__.__name__))
    parse_train_corpus = \
        property(lambda self: self.__err_hideattr('parse_train_corpus'))
    backup = \
        property(lambda self: self.__err_hideattr(self, 'backup'))
    restore = \
        property(lambda self: self.__err_hideattr(self, 'restore'))
    _train_init = \
        property(lambda self: self.__err_hideattr(self, '_train_init'))
    _train_eval = \
        property(lambda self: self.__err_hideattr(self, '_train_eval'))
    _train_done = \
        property(lambda self: self.__err_hideattr(self, '_train_done'))

    def __init__(self):
        super().__init__()
        self._model = None
        self._ds = None

    def load_train_corpus(self, corpus, append=False, test=None, seed=None):
        """Loads the train corpus.

        Args:

        **corpus**: a name of the file in *CoNLL-U* format or a list/iterator
        of sentences in *Parsed CoNLL-U*.

        **append** (`bool`): whether to add **corpus** to the already loaded
        one(s).

        **test** (`float`): if not `None`, **corpus** will be shuffled and a
        specified part of it stored as a test corpus.

        **seed** (`int`): init value for the random number generator if you
        need reproducibility. Used only if test is not `None`.
        """
        args, kwargs = junky.get_func_params(BaseTagger.load_train_corpus,
                                             locals())
        super().load_train_corpus(*args, parse=False, **kwargs)

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
    def _get_filenames(name):
        return tuple(os.path.join(name, x)
                         for x in [_MODEL_CONFIG_FN, _MODEL_FN,
                                   _DATASETS_CONFIG_FN, _DATASETS_FN,
                                   _CDICT_FN])

    @staticmethod
    def _prepare_corpus(corpus, fields, tags_to_remove=None):
        return junky.extract_conllu_fields(
            junky.conllu_remove(corpus, remove=tags_to_remove),
            fields=fields
        )

    def _create_dataset(
        self, sentences, word_emb_type=None, word_emb_path=None,
        word_emb_model_device=None, word_transform_kwargs=None,
        word_next_emb_params=None, with_chars=False, tags=None,
        labels=None, for_self=True, log_file=LOG_FILE
    ):
        ds = FrameDataset()

        if word_emb_type is not None:
            x = WordEmbeddings.create_dataset(
                sentences, emb_type=word_emb_type, emb_path=word_emb_path,
                emb_model_device=word_emb_model_device,
                transform_kwargs=word_transform_kwargs,
                next_emb_params=word_next_emb_params,
                loglevel=0 if log_file is None else
                         1 if log_file == sys.stdin else
                         2
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
            if isinstance(labels[0], tuple):
                y = TokenDataset(labels, pad_token='<PAD>', transform=True,
                                 keep_empty=False)
                ds.add('y', y, with_lens=False)
            else:
                y = LabelDataset(labels, transform=True, keep_empty=False)
                ds.add('y', y)

        if for_self:
            self._ds = ds
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
                    ds_, sentences, batch_size=batch_size,
                    loglevel=0 if log_file is None else
                             1 if log_file == sys.stdin else
                             2
                ):
                    ds_.transform(sentences)
            elif typ == 't':
                ds_.transform(tags[int(idx)])
            elif labels and typ == 'y':
                ds_.transform(labels)

    def _transform_collate(self, sentences, tags=None, labels=None, ds=None,
                           batch_size=BATCH_SIZE, log_file=LOG_FILE):
        if ds is None:
            ds = self._ds
        res = []
        for name in self._ds.list():
            ds_ = self._ds.get_dataset(name)
            name_ = name.split('_', maxsplit=1)
            typ, idx = name_[0], name_[1] if len(name_) > 1 else None
            if typ == 'x':
                res_ = WordEmbeddings.transform_collate(
                    ds_, sentences, batch_size=batch_size,
                    loglevel=int(log_file is not None)
                )
                if not res_:
                    res_ = ds_.transform_collate(
                        sentences, batch_size=batch_size,
                        collate_kwargs={'with_lens': False}
                    )
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

    def _save_dataset(self, name, ds=None):
        if not os.path.isdir(name):
            os.mkdir(name)
        if ds is None:
            ds = self._ds
        ds_config_fn, ds_fn = self._get_filenames(name)[2:4]
        config = {}
        for name in ds.list():
            ds_ = ds.get_dataset(name)
            cfg = getattr(ds_, CONFIG_ATTR, None)
            if cfg:
                config[name] = cfg
        with open(ds_config_fn, 'wt', encoding='utf-8') as f:
            print(json.dumps(config, sort_keys=True, indent=4), file=f)
        ds.save(ds_fn, with_data=False)

    def _load_dataset(self, name, device=None, for_self=True,
                      log_file=LOG_FILE):
        ds_config_fn, ds_fn = self._get_filenames(name)[2:4]
        if log_file:
            print('Loading dataset...', end=' ', file=log_file)
            log_file.flush()
        ds = FrameDataset.load(ds_fn)
        with open(ds_config_fn, 'rt', encoding='utf-8') as f:
            config = json.loads(f.read())
            for name, cfg in config.items():
                WordEmbeddings.apply_config(ds.get_dataset(name), cfg,
                                            device=device)
        if log_file:
            print('done.', file=log_file)
        if for_self:
            self._ds = ds
        return ds

    def save(self, name, log_file=LOG_FILE):
        """Saves the internal state of the tagger.

        Args:

        **name**: a name to save with.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method creates a directory **name** that contains 5 files: two for
        the tagger's model (`model.config.json` and `model.pt`) and two for
        its dataset (`ds.config.json` and `ds.pt`). The 5th file
        (`cdict.pickle`) is an internal state of
        [`corpuscula.CorpusDict`](https://github.com/fostroll/corpuscula/blob/master/doc/README_CDICT.md)
        object that is used by the tagger as a helper.

        `*.config.json` files contain parameters for creation of the objects.
        They are editable, but you are allowed to change only the device name.
        Any other changes most likely won't allow the tagger to load.
        """
        assert self._ds, "ERROR: The tagger doesn't have a dataset to save"
        assert self._model, "ERROR: The tagger doesn't have a model to save"
        self._save_dataset(name)
        model_config_fn, model_fn, _, _, cdict_fn = \
            self._get_filenames(name)
        self._model.save_config(model_config_fn, log_file=log_file)
        self._model.save_state_dict(model_fn, log_file=log_file)
        self._save_cdict(cdict_fn)

    def load(self, model_class, name, device=None, dataset_device=None,
             log_file=LOG_FILE):
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **model_class**: a class of the model used for prediction. Must be
        descendant of `BaseTaggerModel` class.

        **name** (`str`): name of the previously saved internal state.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        override its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        self._load_dataset(name, device=dataset_device, log_file=log_file)
        model_config_fn, model_fn, _, _, cdict_fn = \
            self._get_filenames(name)
        self._model = model_class.create_from_config(
            model_config_fn, state_dict_f=model_fn, device=device,
            log_file=log_file
        )
        self._load_cdict(cdict_fn, log_file=log_file)

    @staticmethod
    def _normalize_field_names(names):
        res, tostr = [], False
        if names:
            if isinstance(names, str):
                names, tostr = [names], True
            for name in names:
                num_colons = name.count(':')
                if num_colons == 0:
                    name += '::' + NONE_TAG
                elif num_colons == 1:
                    name += ':' + NONE_TAG
                elif num_colons == 2 and name.endswith(':'):
                    name += NONE_TAG
                res.append(name)
        return res[0] if tostr else res

    def predict(self, field, add_fields, corpus, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE):
        """Predicts tags in the specified fields for the corpus.

        Args:

        **field** and **add_field** must be the same that were used in the
        `.train()` method.

        **corpus**: a corpus which will be used for feature extraction and
        predictions. May be either a name of the file in *CoNLL-U* format or
        list/iterator of sentences in *Parsed CoNLL-U*.

        **with_orig** (`bool`): if `True`, instead of only a sequence with
        predicted labels, returns a sequence of tuples where the first element
        is a sentence with predicted labels and the second element is original
        sentence. `with_orig` can be `True` only if `save_to` is `None`.
        Default `with_orig=False`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to process a
        large dataset in pieces ("splits"). Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset. There is no big differences between the variants. Both
        should produce identical results.

        **save_to**: file name where the predictions will be saved.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns corpus with tag predictions in the specified field.
        """
        assert self._ds is not None, \
               "ERROR: The tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert self._model, \
               "ERROR: The tagger doesn't have a model. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'

        field = self._normalize_field_names(field)
        add_fields = self._normalize_field_names(add_fields)

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
                    pred_indices = pred.argmax(-1)
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

    def evaluate(self, field, gold, test=None, feats=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **field** must be the same that was used in the `.train()` method.

        **gold**: a corpus of sentences with actual target values to score the
        tagger on. May be either a name of the file in *CoNLL-U* format or
        list/iterator of sentences in *Parsed CoNLL-U*.

        **test**: a corpus of sentences with predicted target values. If
        `None`, the **gold** corpus will be retagged on-the-fly, and the
        result will be used **test**.

        **feats** (`str|list([str])`): one or several subfields of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **label** (`str`): specific label of the target field to be evaluated,
        e.g. `field='UPOS', label='VERB'` or
        `field='FEATS:Animacy', label='Inan'`. Note that to evaluate key-value
        type fields like `FEATS` or `MISC`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to process a
        large dataset in pieces ("splits"). Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset. There is no big differences between the variants. Both
        should produce identical results.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method prints metrics and returns evaluation accuracy.
        """
        if isinstance(feats, str):
            feats = [feats]
        gold = self._get_corpus(gold, log_file=log_file)
        corpora = zip(self._get_corpus(test, log_file=log_file), gold) \
                      if test else \
                  self.predict(gold, with_orig=True,
                               batch_size=batch_size, split=split,
                               clone_ds=clone_ds, log_file=log_file)
        self._normalize_field_names(field)
        header = field.split(':')[:2]
        if len(header) == 2 and not header[1]:
            header = header[:1]
        field = header[0]
        name = header[1] if len(header) == 2 else None
        header = ':'.join(header)
        if label:
            header += '==' + label
        if log_file:
            print('Evaluating ' + header, file=log_file)

        def compare(gold_label, test_label, n, c, nt, ct, ca, ce, cr):
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
            return n, c, nt, ct, ca, ce, cr

        n = c = nt = ct = ca = ce = cr = ntok = ctok = ntokt = ctokt = 0
        nsent = csent = 0
        for nsent, sentences in enumerate(corpora, start=1):
            csent_ = 1
            for gold_token, test_token in zip(*sentences):
                wform = gold_token['FORM']
                if wform and '-' not in gold_token['ID']:
                    gold_label = gold_token[field]
                    test_label = test_token[field]
                    if name:
                        gold_label = gold_label.get(name)
                        test_label = test_label.get(name)
                    isgold = isinstance(gold_label, dict)
                    istest = isinstance(test_label, dict)
                    if isgold and istest:
                        assert not label, \
                            'ERROR: To evaluate exact label of dict field, ' \
                            "add feat name to field param as '<field:feat>'"
                        ntok += 1
                        if gold_label or test_label:
                            ctok_ = 1
                            for feat in feats if feats else set(
                                [*gold_label.keys(), *test_label.keys()]
                            ):
                                gold_feat = gold_label.get(feat)
                                test_feat = test_label.get(feat)
                                n, c_, nt, ct, ca, ce, cr = \
                                    compare(gold_feat, test_feat,
                                            n, c, nt, ct, ca, ce, cr)
                                if c_ == c:
                                    ctok_ = 0
                                    csent_ = 0 
                                else:
                                    c = c_
                            ntokt += 1
                            ctok += ctok_
                            ctokt += ctok_
                        else:
                            ctok += 1
                    elif not (isgold or istest):
                        n, c_, nt, ct, ca, ce, cr = \
                            compare(gold_label, test_label,
                                    n, c, nt, ct, ca, ce, cr)
                        if c_ == c:
                            csent_ = 0 
                        else:
                            c = c_
                    else:
                        raise TypeError(
                            'Inconsistent field types in gold and test '
                            'corpora'
                        )
            csent += csent_
        if test:
            try:
                next(gold)
            except StopIteration:
                pass
        if log_file:
            if nsent <= 0:
                print('Nothing to do!', file=log_file)
            else:
                sp = ' ' * (len(header) - 2)
                print(header + ' total: '
                    + ('{} tokens, {} tags'.format(ntok, nt) if ntok else
                       '{}'.format(nt)), file=log_file)
                print(sp + ' correct: '
                    + ('{} tokens, {} tags'.format(ctok, ct) if ntok else
                       '{}'.format(ct)), file=log_file)
                print(sp + '   wrong: '
                    + ('{} tokens, {} tags'.format(ntok - ctok, nt - ct)
                           if ntok else
                      '{}'.format(nt - ct))
                    + (' [{} excess / {} absent{}]'
                           .format(ce, ca, '' if label else
                                           ' / {} wrong type'.format(cr))
                           if cr != nt - ct else
                       ''), file=log_file)
                print(sp + 'Accuracy: {}{}'
                               .format('{} / '.format(ctokt / ntokt
                                                          if ntokt > 0 else
                                                      1.)
                                           if ntok else
                                       '',
                                       ct / nt if nt > 0 else 1.),
                      file=log_file)
                if nt != n or ntokt != ntok:
                    print('[Total accuracy: {}{}]'
                              .format('{} / '.format(ctok / ntok
                                                         if ntok > 0 else
                                                     1.)
                                          if ntok else
                                      '',
                                      c / n if n > 0 else 1.), file=log_file)
                print('[By sentence accuracy: {}]'
                          .format('{}'.format(csent / nsent)), file=log_file)
        return ct / nt if nt > 0 else 1.

    def train(self, field, add_fields, model_class, tag_emb_names, save_as,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_path=None,
              word_emb_model_device=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              seed=None, log_file=LOG_FILE, **model_kwargs):
        """Creates and trains the tagger model.

        We assume all positional argumets but **save_as** are for internal use
        only and should be hide in descendant classes.

        During training, the best model is saved after each successful epoch.

        *Training's args*:

        **field** (`str`): the name of the field which needs to be predicted
        by the training tagger. May contain up to 3 elements, separated by a
        colon (`:`). Format is:
        `'<field name>:<feat name>:<replacement for None>'`. The replacement
        is used during the training time as a filler for a fields without a
        value for that we could predict them, too. In the *CoNLL-U* format the
        replacer is `'_'` sign, so we use it, too, as a default replacement.
        You'll hardly have a reason to change it. Examples:<br/>
        `'UPOS'` - predict the *UPOS* field;<br/>
        `'FEATS:Animacy'` - predict only the *Animacy* feat of the *FEATS*
        field;<br/>
        `'FEATS:Animacy:_O'` - likewise the above, but if feat value is
        `None`, it will be replaced by `'_O'` during training;<br/>
        `'XPOS::_O'` - predict the *XPOS* field and use `'_O'` as replacement
        for `None`.

        **add_fields** (`None|str|list([str])`): any auxiliary fields to use
        with the *FORM* field for predictions. If `None`, only *FORM* field is
        used for predictions. To use additional fields for predicitons, pass a
        field name (e.g. `'UPOS'`) or a list of field names (e.g. `['UPOS',
        'LEMMA']`). These fields are included in the dataset. The format of
        each element of **add_fields** is equal to the **field** format.

        **model_class**: a class of the model used for prediction. Must be
        descendant of `BaseTaggerModel` class.

        **tag_emb_names** (`str|list([str])`): prefixes of the model args,
        using instead of `tag_emb_params` `dict` of the `BaseTaggerModel`
        class. Each name refers to the corresponding field in the
        **add_fields** arg. You have to look into sources of the descendant
        classess included to the project if you really want to make sense of
        it.

        **save_as** (`str`): the name using for save. Refer to the `.save()`
        method's help for the broad definition (see the **name** arg there).

        **device**: device for the model. E.g.: 'cuda:0'.

        **epochs** (`int`): number of epochs to train. If `None` (default),
        train until `bad_epochs` is met, but no less than `min_epochs`.

        **min_epochs** (`int`): minimum number of training epochs. Default is
        `0`

        **bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
        when selected **control_metric** is became not better) in a row.
        Default is `5`.

        **batch_size** (`int`): number of sentences per batch. For training,
        default `batch_size=32`.

        **control_metric** (`str`): metric to control training. Any that is
        supported by the `junky.train()` method. In the moment it is:
        'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **max_grad_norm** (`float`): gradient clipping parameter, used with
        `torch.nn.utils.clip_grad_norm_()`.

        **tags_to_remove** (`dict({str: str})|dict({str: list([str])})`):
        tags, tokens with those must be removed from the corpus. It's a `dict`
        with field names as keys and with value you want to remove. Applied
        only to fields with atomic values (like UPOS). This argument may be
        used, for example, to remove some infrequent or just excess tags from
        the corpus. Note, that we remove the tokens from the train corpus as a
        whole, not just replace those tags to `None`.

        *Word embedding params*:

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v') embedding
        types.

        **word_emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert'). `None` means
        **word_emb_model_device** = **device**

        **word_emb_path** (`str`): path to word embeddings storage.

        **word_emb_tune_params**: parameters for word embeddings finetuning.
        For now, only BERT embeddings finetuning is supported with
        `mordl.WordEmbeddings.bert_tune()`. So, **word_emb_tune_params** is a
        `dict` of keyword args for this method. You can replace any except
        `test_data`.

        **word_transform_kwargs** (`dict`): keyword arguments for
        `.transform()` method of the dataset created for sentences to word
        embeddings conversion. See the `.transform()` method of
        `junky.datasets.BertDataset` for the the description of the
        parameters.

        **word_next_emb_params**: if you want to use several different
        embedding models at once, pass parameters of the additional model as a
        dictionary with keys `(emb_path, emb_model_device, transform_kwargs)`;
        or a list of such dictionaries if you need more than one additional
        model.

        *Other options*:

        **seed** (`int`): init value for the random number generator if you
        need reproducibility.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        **model_kwargs**: keyword arguments for the model creating. Will be
        passed as is to the **model_class** constructor.

        The method returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if isinstance(tag_emb_names, str):
            tag_emb_names = [tag_emb_names]

        field = self._normalize_field_names(field)
        header = field.split(':')[:2]
        if len(header) == 2 and not header[1]:
            header = header[:1]
        header = ':'.join(header)
        bert_header = header.lower().replace(':', '-') + '_'
        fields = self._normalize_field_names(
            [] if add_fields is None else \
            [add_fields] if isinstance(add_fields, str) else \
            add_fields
        )
        fields.append(field)

        if log_file:
            print('=== {} TAGGER TRAINING PIPELINE ==='.format(header),
                  file=log_file)

        # 1. Prepare corpora
        train = self._prepare_corpus(
            self._train_corpus, fields=fields,
            tags_to_remove=tags_to_remove
        )
        test = self._prepare_corpus(
            self._test_corpus, fields=fields,
            tags_to_remove=tags_to_remove
        ) if self._test_corpus is not None else None

        # 2. Tune embeddings
        if word_emb_model_device is None:
            word_emb_model_device = device

        def tune_word_emb(emb_type, emb_path, emb_model_device=None,
                          emb_tune_params=None):
            if emb_tune_params is True:
                emb_tune_params = {}
            elif isinstance(emb_tune_params, str):
                emb_tune_params = {'model_name': emb_tune_params}
            if isinstance(emb_tune_params, dict):
                emb_tune_params = dict(emb_tune_params.items())
                if emb_type == 'bert':
                    if 'test_data' not in emb_tune_params and test:
                        emb_tune_params['test_data'] = test[0], test[-1]
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
                    raise ValueError("ERROR: Tune method for '{}' embeddings "
                                         .format(emb_type)
                                   + 'is not implemented')
            elif emb_tune_params not in [None, False]:
                raise TypeError(
                    'ERROR: emb_tune_params is of incorrect type. '
                    'It can be either dict, str, bool or None'
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
        log_file_ = sys.stderr if log_file else None
        ds_train = self._create_dataset(
            train[0],
            word_emb_type=word_emb_type, word_emb_path=word_emb_path,
            word_emb_model_device=word_emb_model_device,
            word_transform_kwargs=word_transform_kwargs,
            word_next_emb_params=word_next_emb_params,
            with_chars=model_kwargs.get('rnn_emb_dim') \
                    or model_kwargs.get('cnn_emb_dim'),
            tags=train[1:-1], labels=train[-1], for_self=False,
            log_file=log_file_)
        self._save_dataset(save_as, ds=ds_train)
        if test:
            ds_test = ds_train.clone(with_data=False)
            self._transform(test[0], tags=test[1:-1], labels=test[-1],
                            ds=ds_test, log_file=log_file_)
        else:
            ds_test = None

        model_config_fn, model_fn, _, _, cdict_fn = \
            self._get_filenames(save_as)
        self._save_cdict(cdict_fn)

        # 4. Create model
        if log_file:
            print('\nMODEL CREATION', file=log_file)
        ds_ = ds_train.get_dataset('y')
        model_args = [len(ds_.transform_dict)]
        if hasattr(ds_, 'pad'):
            model_kwargs['labels_pad_idx'] = ds_.pad
        if word_emb_type:
            ds_ = ds_train.get_dataset('x')
            model_kwargs['vec_emb_dim'] = ds_.vec_size
        if model_kwargs.get('rnn_emb_dim') or model_kwargs.get('cnn_emb_dim'):
            ds_ = ds_train.get_dataset('x_ch')
            model_kwargs['alphabet_size'] = len(ds_.transform_dict)
            model_kwargs['char_pad_idx'] = ds_.pad
        if tag_emb_names:
            ds_list = ds_train.list()
            names = iter(tag_emb_names)
            for name in ds_train.list():
                if name.startswith('t_'):
                    ds_ = ds_train.get_dataset(name)
                    name = next(names)
                    emb_dim = model_kwargs[name + '_emb_dim']
                    if emb_dim:
                        model_kwargs[name + '_num'] = len(ds_.transform_dict)
                        model_kwargs[name + '_pad_idx'] = ds_.pad
        model, criterion, optimizer, scheduler = \
            model_class.create_model_for_train(*model_args, **model_kwargs)
        if device:
            model.to(device)
        model.save_config(model_config_fn, log_file=log_file)

        # 5. Train model
        if log_file:
            print('\nMODEL TRAINING', file=log_file)
        def best_model_backup_method(model, model_score):
            if log_file:
                print('new maximum score {:.8f}'.format(model_score),
                      file=log_file)
            model.save_state_dict(model_fn, log_file=log_file)
        res_ = junky.train(
            None, model, criterion, optimizer, scheduler,
            best_model_backup_method, datasets=(ds_train, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric=control_metric,
            max_grad_norm=max_grad_norm,
            with_progress=log_file is not None, log_file=log_file
        )
        best_epoch, best_score = res_['best_epoch'], res_['best_score']
        res = {x: y for x, y in res_.items()
                        if x not in ['best_epoch', 'best_score']}

        # 6. Tune model
        if log_file:
            print('\nMODEL TUNING', file=log_file)
        model.load_state_dict(model_fn, log_file=log_file)
        criterion, optimizer, scheduler = model.adjust_model_for_tune()
        res_= junky.train(
            None, model, criterion, optimizer, scheduler,
            best_model_backup_method, datasets=(ds_train, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric=control_metric,
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

        del model, ds_train, ds_test

        if log_file:
            print('\n=== {} TAGGER TRAINING HAS FINISHED ===\n'
                      .format(header), file=log_file)
            print(("Use the `.load('{}')` method to start working "
                   'with the {} tagger.').format(save_as, header),
                      file=log_file)

        return res


def conll18_ud_eval(gold_file, system_file, verbose=True, counts=False):
    """A wrapper for the official
    [CoNLL18 UD Shared Task](https://universaldependencies.org/conll18/results.html)
    evaluation script.

    Args:

    **gold_file**: Name of the CoNLL-U file with the gold data.

    **system_file**: Name of the CoNLL-U file with the predicted.

    **verbose** (`bool`): Print all metrics.

    **counts** (`bool`): Print raw counts of correct/gold/system/aligned words
    instead of prec/rec/F1 for all metrics.

    If `verbose=False`, only the official CoNLL18 UD Shared Task evaluation
    metrics are printed.

    If `verbose=True` (default), more metrics are printed (as precision,
    recall, F1 score, and in case the metric is computed on aligned words
    also accuracy on these):
    - Tokens: how well do the gold tokens match system tokens.
    - Sentences: how well do the gold sentences match system sentences.
    - Words: how well can the gold words be aligned to system words.
    - UPOS: using aligned words, how well does UPOS match.
    - XPOS: using aligned words, how well does XPOS match.
    - UFeats: using aligned words, how well does universal FEATS match.
    - AllTags: using aligned words, how well does UPOS+XPOS+FEATS match.
    - Lemmas: using aligned words, how well does LEMMA match.
    - UAS: using aligned words, how well does HEAD match.
    - LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes)
    match.
    - CLAS: using aligned words with content DEPREL, how well does
    HEAD+DEPREL(ignoring subtypes) match.
    - MLAS: using aligned words with content DEPREL, how well does
    HEAD+DEPREL(ignoring subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS)
    match.
    - BLEX: using aligned words with content DEPREL, how well does
    HEAD+DEPREL(ignoring subtypes)+LEMMAS match.

    If `count=True`, raw counts of correct/gold_total/system_total/aligned
    words are printed instead of precision/recall/F1/AlignedAccuracy for all
    metrics."""
    argv = sys.argv
    sys.argv[1:] = [gold_file, system_file]
    if verbose:
        sys.argv.append('-v')
    if counts:
        sys.argv.append('-c')
    _conll18_ud_eval()
    sys.argv = argv
