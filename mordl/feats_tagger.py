# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides joint and separate key-value type field tagger classes.
"""
from copy import deepcopy
from collections import OrderedDict
from difflib import get_close_matches
import itertools
import json
from junky import clear_tqdm, get_func_params
from mordl import FeatTagger
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, CONFIG_EXT, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel
from typing import Iterator


class FeatsJointTagger(BaseTagger):
    """
    A class for prediction a content ot a key-value type field. Joint
    implementation (predict all the content of the field at once).

    Args:

    **field** (`str`): a name of the *CoNLL-U* key-value type field, content
    of which needs to be predicted. With the tagger, you can predict only
    key-value type fields, like FEATS.
    """
    def __init__(self, field='FEATS'):
        super().__init__()
        self._field = field

    def load(self, name, device=None, dataset_device=None, log_file=LOG_FILE):
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **name** (`str`): name of the previously saved internal state.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        override its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        args, kwargs = get_func_params(FeatsJointTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts feature keys and values in the FEATS field of the corpus.

        Args:

        **corpus**: a corpus which will be used for feature extraction and
        predictions. May be either a name of the file in *CoNLL-U* format or
        list/iterator of sentences in *Parsed CoNLL-U*.

        **with_orig** (`bool`): if `True`, instead of only a sequence with
        predicted labels, returns a sequence of tuples where the first element
        is a sentence with predicted labels and the second element is the
        original sentence. `with_orig` can be `True` only if `save_to` is
        `None`. Default `with_orig=False`.

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

        Returns corpus with feature keys and values predicted in the FEATS
        field.
        """
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(FeatsJointTagger.predict, locals())
        kwargs['save_to'] = None

        def process(corpus):
            for sentence in corpus:
                sentence_ = sentence[0] if with_orig else sentence
                if isinstance(sentence_, tuple):
                    sentence_ = sentence_[0]
                for token in sentence_:
                    token[self._field] = OrderedDict(
                        [(x, y) for x, y in [
                            x.split('=')
                                for x in token[self._field].split('|')
                        ]]
                    ) if token[self._field] else OrderedDict()
                yield sentence

        corpus = process(
            super().predict(self._field, 'UPOS', *args, **kwargs)
        )
        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, gold, test=None, feats=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: a corpus of sentences with actual target values to score the
        tagger on. May be either a name of the file in *CoNLL-U* format or
        list/iterator of sentences in *Parsed CoNLL-U*.

        **test**: a corpus of sentences with predicted target values. If
        `None`, the **gold** corpus will be retagged on-the-fly, and the
        result will be used **test**.

        **feats** (`str|list([str])`): one or several feature names of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **label** (`str`): specific label of the target feature value to be
        evaluated, e.g. `label='Inan'`. If you specify a value here, you must
        also specify the feature name as **feats** param (e.g.:
        `feats=`'Animacy'`). Note, that in that case the param **feats** must
        contain only one feature name.

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
        assert not label or feats, \
            'ERROR: To evaluate the exact label you must specify its ' \
            'feat, too'
        assert not label or not feats \
                         or isinstance(feats, str) or len(feats) == 1, \
            'ERROR: To evaluate the exact label you must specify its own ' \
            'feat only'
        args, kwargs = get_func_params(FeatsJointTagger.evaluate, locals())
        field = self._field
        if label:
            del kwargs['feats']
            field += ':' + (feats if isinstance(feats, str) else feats[0])
        return super().evaluate(field, *args, **kwargs)

    def train(self, save_as,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=200, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=3, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
        """Creates and trains a key-value type field tagger model.

        During training, the best model is saved after each successful epoch.

        *Training's args*:

        **save_as** (`str`): the name used for save. Refer to the `.save()`
        method's help for the broad definition (see the **name** arg there).

        **device**: device for the model. E.g.: 'cuda:0'.

        **epochs** (`int`): number of epochs to train. If `None` (default),
        train until `bad_epochs` is met, but no less than `min_epochs`.

        **min_epochs** (`int`): minimum number of training epochs. Default is
        `0`

        **bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
        during which the selected **control_metric** does not improve) in a
        row. Default is `5`.

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

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v')
        embedding types.

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
        embedding models at once, pass the parameters of the additional model
        as a dictionary with keys
        `(emb_path, emb_model_device, transform_kwargs)`; or a list of such
        dictionaries if you need more than one additional model.

        *Model hyperparameters*:

        **rnn_emb_dim** (`int`): character RNN (LSTM) embedding
        dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
        `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
        `cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
        **cnn_emb_dim**.

        **upos_emb_dim** (`int`): auxiliary embedding dimensionality for UPOS
        labels. Default `upos_emb_dim=200`.

        **emb_out_dim** (`int`): output embedding dimensionality. Default
        `emb_out_dim=512`.

        **lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
        `lstm_hidden_dim=256`.

        **lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
        `lstm_layers=3`.

        **lstm_do** (`float`): dropout between LSTM layers. Only relevant, if
        `lstm_layers` > `1`.

        **bn1** (`bool`): whether batch normalization layer should be applied
        after the embedding layer. Default `bn1=True`.

        **do1** (`float`): dropout rate after the first batch normalization
        layer `bn1`. Default `do1=.2`.

        **bn2** (`bool`): whether batch normalization layer should be applied
        after the linear layer before LSTM layer. Default `bn2=True`.

        **do2** (`float`): dropout rate after the second batch normalization
        layer `bn2`. Default `do2=.5`.

        **bn3** (`bool`): whether batch normalization layer should be applied
        after the LSTM layer. Default `bn3=True`.

        **do3** (`float`): dropout rate after the third batch normalization
        layer `bn3`. Default `do3=.4`.

        *Other options*:

        **seed** (`int`): init value for the random number generator if you
        need reproducibility.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        args, kwargs = get_func_params(FeatsJointTagger.train, locals())

        [x.update({self._field: '|'.join('='.join((y, x[self._field][y]))
                                    for y in sorted(x[self._field]))})
             for x in self._train_corpus for x in x]

        [x.update({self._field: '|'.join('='.join((y, x[self._field][y]))
                                    for y in sorted(x[self._field]))})
             for x in self._test_corpus for x in x]

        key_vals = set(x[self._field] for x in self._train_corpus for x in x)
        [None if x[self._field] in key_vals else
         x.update({self._field: [*get_close_matches(x[self._field],
                                                    key_vals, n=1), ''][0]})
             for x in self._test_corpus for x in x]

        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)


class FeatsSeparateTagger(BaseTagger):
    """
    A class for prediction a content ot a key-value type field. Separate
    implementation (for each feature, creates its own tagger).

    Args:

    **field** (`str`): a name of the *CoNLL-U* key-value type field, content
    of which needs to be predicted. With the tagger, you can predict only
    key-value type fields, like FEATS.
    """
    def __init__(self, field='FEATS'):
        super().__init__()
        self._field = field
        self._feats = {}

    def save(self, name, log_file=LOG_FILE):
        """Saves the config of tagger with paths to the separate taggers'
        storages.

        Args:

        **name**: a name to save with.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        if not name.endswith(CONFIG_EXT):
            name += CONFIG_EXT
        with open(name, 'wt', encoding='utf-8') as f:
            print(json.dumps({x: y[0] if isinstance(y, dict) else y
                                  for x, y in self._feats.items()},
                             sort_keys=True, indent=4), file=f)
        if log_file:
            print('Config saved', file=log_file)

    def load(self, name, log_file=LOG_FILE):
        """Reads previously saved config and loads all taggers that it
        contain.

        Args:

        **name**: a name of the tagger's config.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        if not name.endswith(CONFIG_EXT):
            name += CONFIG_EXT
        with open(name, 'rt', encoding='utf-8') as f:
            self._feats = json.loads(f.read())
        if log_file:
            print('### Loading {} tagger:'.format(self._field), file=log_file)
        for feat in sorted(self._feats):
            if log_file:
                print('\n--- {}:'.format(feat), file=log_file)
            name_ = self._feats[feat]
            tagger = FeatTagger(feat)
            tagger.load(name_)
            self._feats[feat] = [name, tagger]
        if log_file:
            print('\n{} tagger have been loaded ###'.format(self._field),
                  file=log_file)

    def predict(self, corpus, feats=None, remove_excess_feats=True,
                with_orig=False, batch_size=BATCH_SIZE, split=None,
                clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts feature keys and values in the FEATS field of the corpus.

        Args:

        **corpus**: a corpus which will be used for feature extraction and
        predictions. May be either a name of the file in *CoNLL-U* format or
        list/iterator of sentences in *Parsed CoNLL-U*.

        **feats** (`str|list([str)`): exact features to be predicted.

        **remove_excess_feats** (`bool`): if `True` (default), the tagger
        removes all unrelevant features from predicted field ("unrelevant"
        means, that the tagger don't have a models for them). For example, if
        you trained the tagger only for "Case" and "Gender" features, the
        tagger predict only them (or, only one of them, if you specify it in
        **feats** field) and remove all the rest. If
        `remove_excess_feats=False`, all unrelevant feats will be stayed
        intact.

        **with_orig** (`bool`): if `True`, instead of only a sequence with
        predicted labels, returns a sequence of tuples where the first element
        is a sentence with predicted labels and the second element is the
        original sentence. `with_orig` can be `True` only if `save_to` is
        `None`. Default `with_orig=False`.

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

        Returns corpus with feature keys and values predicted in the FEATS
        field.
        """
        args, kwargs = get_func_params(FeatsSeparateTagger.predict, locals())
        del kwargs['feats']
        del kwargs['remove_excess_feats']

        if feats is None:
            feats = self._feats
        else:
            if isinstance(feats, str):
                feats = [feats]
            unknown_feats = []
            for feat in sorted(feats):
                if feat not in self._feats:
                    unknown_feats.append(feat)
            assert unknown_feats, \
                'ERROR: feats {} are unknown for the tagger' \
                    .format(unknown_feats)

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

                if remove_excess_feats:
                    for sentence in corpus_:
                        for token in sentence[0] \
                                         if isinstance(sentence, tuple) else \
                                     sentence:
                            token[self._field] = OrderedDict(
                                (x, y) for x, y in token[self._field].items()
                                           if x in feats
                            )

                res_corpus_ = deepcopy(corpus_) if with_orig else corpus_

                for attrs in self._feats.values():
                    tagger = attrs[1] \
                                 if isinstance(attrs, list) else \
                             None
                    assert isinstance(tagger, FeatTagger), \
                        'ERROR: Model is not loaded. Use the .load() ' \
                        'method prior'
                    res_corpus_ = tagger.predict(res_corpus_, **kwargs)

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

    def evaluate(self, gold, test=None, feats=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: a corpus of sentences with actual target values to score the
        tagger on. May be either a name of the file in *CoNLL-U* format or a
        list/iterator of sentences in *Parsed CoNLL-U*.

        **test**: a corpus of sentences with predicted target values. If
        `None`, the **gold** corpus will be retagged on-the-fly, and the
        result will be used as **test**.

        **feats** (`str|list([str])`): one or several feature names of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **label** (`str`): specific label of the target feature value to be
        evaluated, e.g. `label='Inan'`. If you specify a value here, you must
        also specify the feature name as **feats** param (e.g.:
        `feats='Animacy'`). Note, that in that case the param **feats** must
        contain only one feature name.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to process a
        large dataset in pieces ("splits"). Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset. There is no big difference between the variants. Both
        should produce identical results.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method prints metrics and returns evaluation accuracy.
        """
        assert not label or feats, \
            'ERROR: To evaluate the exact label you must specify its ' \
            'feat, too'
        assert not label or not feats \
                         or isinstance(feats, str) or len(feats) == 1, \
            'ERROR: To evaluate the exact label you must specify its own ' \
            'feat only'
        args, kwargs = get_func_params(FeatsSeparateTagger.evaluate, locals())
        field = self._field
        if label:
            del kwargs['feats']
            field += ':' + (feats if isinstance(feats, str) else feats[0])
        return super().evaluate(field, *args, **kwargs)

    def train(self, save_as, feats=None,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path_suffix=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=200, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=3, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
        """Creates and trains a separate feature tagger model.

        *Training's args*:

        **save_as** (`str`): the name used for save. Refer to the `.save()`
        method's help for the broad definition (see the **name** arg there).

        **feats** (`str|list([str])`): train model only for predicting one or
        several subfields of the key-value type fields like FEATS or MISC.
        E.g., for tagger created with `field='FEATS'` argument, allowed values
        for **feats** are: 'Animacy', ['Case', 'Polarity', 'Tense'] etc.

        **device**: device for the model. E.g.: 'cuda:0'.

        **epochs** (`int`): number of epochs to train. If `None` (default),
        train until `bad_epochs` is met, but no less than `min_epochs`.

        **min_epochs** (`int`): minimum number of training epochs. Default is
        `0`

        **bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
        during which the selected **control_metric** does not improve) in a
        row. Default is `5`.

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

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v')
        embedding types.

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
        embedding models at once, pass the parameters of the additional model
        as a dictionary with keys
        `(emb_path, emb_model_device, transform_kwargs)`; or a list of such
        dictionaries if you need more than one additional model.

        *Model hyperparameters*:

        **rnn_emb_dim** (`int`): character RNN (LSTM) embedding
        dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
        `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
        `cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
        **cnn_emb_dim**.

        **upos_emb_dim** (`int`): auxiliary embedding dimensionality for UPOS
        labels. Default `upos_emb_dim=200`.

        **emb_out_dim** (`int`): output embedding dimensionality. Default
        `emb_out_dim=512`.

        **lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
        `lstm_hidden_dim=256`.

        **lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
        `lstm_layers=3`.

        **lstm_do** (`float`): dropout between LSTM layers. Only relevant, if
        `lstm_layers` > `1`.

        **bn1** (`bool`): whether batch normalization layer should be applied
        after the embedding layer. Default `bn1=True`.

        **do1** (`float`): dropout rate after the first batch normalization
        layer `bn1`. Default `do1=.2`.

        **bn2** (`bool`): whether batch normalization layer should be applied
        after the linear layer before LSTM layer. Default `bn2=True`.

        **do2** (`float`): dropout rate after the second batch normalization
        layer `bn2`. Default `do2=.5`.

        **bn3** (`bool`): whether batch normalization layer should be applied
        after the LSTM layer. Default `bn3=True`.

        **do3** (`float`): dropout rate after the third batch normalization
        layer `bn3`. Default `do3=.4`.

        *Other options*:

        **seed** (`int`): init value for the random number generator if you
        need reproducibility.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        args, kwargs = get_func_params(FeatsSeparateTagger.train, locals())
        del kwargs['feats']
        del kwargs['word_emb_path_suffix']

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

            save_as_ = '{}-{}'.format(save_as, feat.lower())
            self._feats[feat] = save_as_

            tagger = FeatTagger(self._field + ':' + feat)
            tagger._train_corpus, tagger._test_corpus = \
                self._train_corpus, self._test_corpus
            if word_emb_path_suffix:
                kwargs['word_emb_path'] = \
                    '{}-{}_{}'.format(self._field.lower(), feat.lower(),
                                      word_emb_path_suffix)
            res[feat] = tagger.train(save_as_, **kwargs)

            del tagger

        self.save(save_as, log_file=log_file)
        if log_file:
            print('\n###### {} TAGGER TRAINING HAS FINISHED ######\n'
                      .format(self._field), file=log_file)
            print(("Now, check the separate {} models' and datasets' "
                   'config files and consider to change some device names '
                   'to be able load all the models jointly. You can find '
                   'the separate models\' list in the "{}" config file. '
                   "Then, use the `.load('{}')` method to start working "
                   'with the {} tagger.').format(self._field, save_as,
                                                 save_as, self._field),
                      file=log_file)
        return res
