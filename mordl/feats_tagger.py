# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides joint and separate multiple-tag FEAT tagger classes.
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
    A class for joint multiple-tag FEAT feature tagging.
    """

    def __init__(self, field='FEATS'):
        super().__init__()
        self._field = field

    def load(self, name, device=None, dataset_device=None, log_file=LOG_FILE):
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **name** (`str`): name of the internal state previously saved.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        overrride its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        args, kwargs = get_func_params(FeatsJointTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the specified fields for the corpus.

        Args:

        **corpus**: input corpus which will be used for feature extraction and
        predictions.

        **with_orig** (`bool`): if `True`, instead of only a sequence with
        predicted labels, returns a sequence of tuples where the first element
        is a sentence with predicted labels and the second element is original
        sentence labels. `with_orig` can be `True` only if `save_to` is
        `None`. Default `with_orig=False`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to split a
        large dataset into several parts. Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset.

        **save_to**: directory where the predictions will be saved.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns corpus with tag predictions in the specified field.
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
        """Evaluate predicitons on the development test set.

        Args:

        **gold** (`tuple(<sentences> <labels>)`): corpus with actual target
        tags.

        **test** (`tuple(<sentences> <labels>)`): corpus with predicted target
        tags. If `None`, predictions will be created on-the-fly based on the
        `gold` corpus.

        **feats** (`str|list([str])`): one or several subfields of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **label** (`str`): specific label of the target field to be evaluated,
        e.g. `field='UPOS'`, `label='VERB'` or `field='FEATS:Animacy'`,
        `label='Inan'`. Note that to evaluate key-value type fields like
        `FEATS` or `MISC`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to split a
        large dataset into several parts. Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Prints metrics and returns evaluation accuracy.
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
              upos_emb_dim=None, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
        """Creates and trains a feature tagger model.

        We assume all positional argumets but **save_as** are for internal use
        only and should be hidden in descendant classes.

        Args:

        **save_as** (`str`): the name of the tagger using for save. There 4
        files will be created after training: two for tagger's model (config
        and state dict) and two for the dataset (config and the internal
        state). All file names are used **save_as** as prefix and their
        endings are: `.config.json` and `.pt` for the model; `_ds.config.json`
        and `_ds.pt` for the dataset.

        **device**: device for the model. E.g.: 'cuda:0'.

        **epochs** (`int`): number of epochs to train. If `None`, train until
        `bad_epochs` is met, but no less than `min_epochs`.

        **min_epochs** (`int`): minimum number of training epochs.

        **bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
        when selected **control_metric** is became not better) in a row.
        Default `bad_epochs=5`.

        **batch_size** (`int`): number of sentences per batch. For training,
        default `batch_size=32`.

        **control_metric** (`str`): metric to control training. Any that is
        supported by the `junky.train()` method. In the moment it is:
        'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **max_grad_norm** (`float`): gradient clipping parameter, used with
        `torch.nn.utils.clip_grad_norm_()`.

        **tags_to_remove** (`list([str])|dict({str: list([str])})`): tags,
        tokens with those must be removed from the corpus. May be a `list` of
        tag names or a `dict` of `{<feat name>: [<feat value>, ...]}`. This
        argument may be used, for example, to remove some infrequent tags from
        the corpus. Note, that we remove the tokens from the train corpus as a
        whole, not just replace those tags to `None`.

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v') embedding
        types.

        **word_emb_path** (`str`): path to word embeddings storage.

        **word_emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert').

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

        **rnn_emb_dim** (`int`): character RNN (LSTM) embedding
        dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
        `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
        `cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
        **cnn_emb_dim**.

        **upos_emb_dim** (`int`): auxiliary UPOS label embedding
        dimensionality. Default `upos_emb_dim=60`.

        **emb_out_dim** (`int`): output embedding dimensionality. Default
        `emb_out_dim=512`.

        **lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
        `lstm_hidden_dim=256`.

        **lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
        `lstm_layers=1`.

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

        **seed** (`int`): init value for the random number generator if you
        need reproducibility.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        args, kwargs = get_func_params(FeatsJointTagger.train, locals())

        [x.update({self._field: find_affixes(x['FORM'], x[self._field])})
             for x in self._train_corpus for x in x]

        [x.update({self._field: find_affixes(x['FORM'], x[self._field])})
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
    A class for separate multiple-tag FEAT feature tagging.
    """

    def __init__(self, field='FEATS'):
        super().__init__()
        self._field = field
        self._feats = {}

    def save(self, name, log_file=LOG_FILE):
        """Saves the internal state of the tagger.

        Args:

        **name**: a name to save with.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        The method creates 4 files for a tagger: two for its model (config and
        state dict) and two for the dataset (config and the internal state).
        All file names started with **name** and their endings are:
        `.config.json` and `.pt` for the model; `_ds.config.json` and `_ds.pt`
        for the dataset.
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
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **name** (`str`): name of the internal state previously saved.

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
        """Predicts tags in the FEATS field.

        Args:

        **corpus**: input corpus which will be used for feature extraction and
        predictions.

        **feats** (`str|list([str)`): exact features to be predicted.

        **remove_excess_feats** (`bool`): if `True`, removes unused feats from
        the corpus. Default `remove_excess_feats=True`.

        **with_orig** (`bool`): if `True`, instead of only a sequence with
        predicted labels, returns a sequence of tuples where the first element
        is a sentence with predicted labels and the second element is original
        sentence labels. `with_orig` can be `True` only if `save_to` is
        `None`. Default `with_orig=False`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to split a
        large dataset into several parts. Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset.

        **save_to**: directory where the predictions will be saved.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns corpus with tag predictions in the specified field.
        """
        args, kwargs = get_func_params(FeatsSeparateTagger.predict, locals())
        del kwargs['feat']
        del kwargs['remove_excess_feats']

        if feat:
            attrs = self._feats[feat]
            tagger = attrs[1] if isinstance(attrs, list) > 1 else None
            assert isinstance(tagger, FeatTagger), \
                'ERROR: Model is not loaded. Use the .load() method prior'
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

                    if remove_excess_feats:
                        for sentence in corpus_:
                            for token in sentence[0] \
                                             if isinstance(sentence,
                                                           tuple) else \
                                         sentence:
                                token[self._field] = \
                                    OrderedDict((x, y)
                                        for x, y in token[self._field].items()
                                            if x in self._feats.keys())

                    res_corpus_ = deepcopy(corpus_) if with_orig else corpus_

                    #if remove_excess_feats:
                    #    for sentence in res_corpus_:
                    #        for token in sentence[0] \
                    #                         if isinstance(sentence,
                    #                                       tuple) else \
                    #                     sentence:
                    #            token[self._field] = OrderedDict()

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
        """valuate predicitons on the development test set.

        Args:

        **gold** (`tuple(<sentences> <labels>)`): corpus with actual target
        tags.

        **test** (`tuple(<sentences> <labels>)`): corpus with predicted target
        tags. If `None`, predictions will be created on-the-fly based on the
        `gold` corpus.

        **feats** (`str|list([str])`): one or several subfields of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **label** (`str`): specific label of the target field to be evaluated,
        e.g. `field='UPOS'`, `label='VERB'` or `field='FEATS:Animacy'`,
        `label='Inan'`. Note that to evaluate key-value type fields like
        `FEATS` or `MISC`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=64`.

        **split** (`int`): number of lines in each split. Allows to split a
        large dataset into several parts. Default `split=None`, i.e. process
        full dataset without splits.

        **clone_ds** (`bool`): if `True`, the dataset is cloned and
        transformed. If `False`, `transform_collate` is used without cloning
        the dataset.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Prints metrics and returns evaluation accuracy.
        """
        assert not label or feats, \
            'ERROR: To evaluate the exact label you must specify its ' \
            'feat, too'
        assert not label or not feats \
                         or isinstance(feats, str) or len(feats) == 1, \
            'ERROR: To evaluate the exact label you must specify its own ' \
            'feat only'
        args, kwargs = get_func_params(FeatsSeparateTagger.evaluate, locals())
        field = self._orig_field
        if label:
            del kwargs['feats']
            field += ':' + (feats if isinstance(feats, str) else feats[0])
        else:
            kwargs['feats'] = \
                sorted([x for x in self._feats.keys()
                              if x not in feats or feats is None])
        return super().evaluate(field, *args, **kwargs)

    def train(self, save_as, feats=None,
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
        """Creates and trains a separate feature tagger model.

        We assume all positional argumets but **save_as** are for internal use
        only and should be hidden in descendant classes.

        Args:

        **save_as** (`str`): the name of the tagger using for save. There 4
        files will be created after training: two for tagger's model (config
        and state dict) and two for the dataset (config and the internal
        state). All file names are used **save_as** as prefix and their
        endings are: `.config.json` and `.pt` for the model; `_ds.config.json`
        and `_ds.pt` for the dataset.

        **feats** (`str|list([str])`): one or several subfields of the
        key-value type fields like `FEATS` or `MISC` to be evaluated.

        **device**: device for the model. E.g.: 'cuda:0'.

        **epochs** (`int`): number of epochs to train. If `None`, train until
        `bad_epochs` is met, but no less than `min_epochs`.

        **min_epochs** (`int`): minimum number of training epochs.

        **bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
        when selected **control_metric** is became not better) in a row.
        Default `bad_epochs=5`.

        **batch_size** (`int`): number of sentences per batch. For training,
        default `batch_size=32`.

        **control_metric** (`str`): metric to control training. Any that is
        supported by the `junky.train()` method. In the moment it is:
        'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **max_grad_norm** (`float`): gradient clipping parameter, used with
        `torch.nn.utils.clip_grad_norm_()`.

        **tags_to_remove** (`list([str])|dict({str: list([str])})`): tags,
        tokens with those must be removed from the corpus. May be a `list` of
        tag names or a `dict` of `{<feat name>: [<feat value>, ...]}`. This
        argument may be used, for example, to remove some infrequent tags from
        the corpus. Note, that we remove the tokens from the train corpus as a
        whole, not just replace those tags to `None`.

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v') embedding
        types.

        **word_emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert').

        **word_emb_path_suffix** (`str`): path suffix to word embeddings
        storage, from full embedding name in the format
        `'<feat>_<word_emb_path_suffix>'`.

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

        **rnn_emb_dim** (`int`): character RNN (LSTM) embedding
        dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
        `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
        `cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
        **cnn_emb_dim**.

        **upos_emb_dim** (`int`): auxiliary UPOS label embedding
        dimensionality. Default `upos_emb_dim=60`.

        **emb_out_dim** (`int`): output embedding dimensionality. Default
        `emb_out_dim=512`.

        **lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
        `lstm_hidden_dim=256`.

        **lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
        `lstm_layers=1`.

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

        **seed** (`int`): init value for the random number generator if you
        need reproducibility.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        args, kwargs = get_func_params(FeatsSeparateTagger.train, locals())
        del kwargs['feats']
        del kwargs['word_emb_path_suffix']

        [x.update({self._field:
                       '|'.join('='.join((y, x[self._field][y]))
                                    for y in sorted(x[self._field]))})
             for x in self._train_corpus for x in x]
        [x.update({self._field:
                       '|'.join('='.join((y, x[self._field][y]))
                                    for y in sorted(x[self._field]))})
             for x in self._test_corpus for x in x]

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
