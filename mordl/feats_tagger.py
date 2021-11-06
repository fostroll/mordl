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
from junky import clear_tqdm, get_func_params, seconds_to_strtime
from mordl import FeatTagger
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, CONFIG_EXT, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel
import time
from typing import Iterator


class FeatsJointTagger(BaseTagger):
    """
    The class for prediction of the content of a key-value type field. The
    joint implementation (predicts all the content of the field at once).

    Args:

    **field** (`str`; default is `FEATS`): the name of the *CoNLL-U* key-value
    type field, content of which needs to be predicted. With this tagger, you
    can predict only key-value type fields, like FEATS.

    **embs** (`dict({str: object}); default is `None`): the `dict` with paths
    to embeddings files as keys and corresponding embedding models as values.
    If the tagger needs to load any embedding model, firstly, the model is
    looked up it in that `dict`.

    During init, **embs** is copied to the `embs` attribute of the creating
    object, and this attribute may be used further to share already loaded
    embeddings with another taggers.
    """
    def __init__(self, field='FEATS', embs=None):
        super().__init__(embs=embs)
        self._field = field

    def load(self, name, device=None,
             dataset_emb_path=None, dataset_device=None, log_file=LOG_FILE):
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **name** (`str`): the name of the previously saved internal state.

        **device** (`str`; default is `None`): the device for the loaded model
        if you want to override the value from the config.

        **dataset_emb_path** (`str`; default is `None`): the path where the
        dataset's embeddings to load from if you want to override the value
        from the config.

        **dataset_device** (`str`; default is `None`): the device for the
        loaded dataset if you want to override the value from the config.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        args, kwargs = get_func_params(FeatsJointTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, use_cdict_coef=False, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE, **_):
        """Predicts feature keys and values in the key-value type field of the
        corpus.

        Args:

        **corpus**: the corpus which will be used for the feature extraction
        and predictions. May be either the name of the file in *CoNLL-U*
        format or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

        **use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
        we use our prediction only. If `True`, we replace our prediction to
        the value returned by the `corpuscula.CorpusDict.predict_<field>()`
        method if its `coef` >= `.99`. Also, you can specify your own
        threshold as the value of the param.

        **with_orig** (`bool`; default is `False`): if `True`, instead of just
        the sequence with predicted labels, return the sequence of tuples
        where the first element is the sentence with predicted labels and the
        second element is the original sentence. **with_orig** can be `True`
        only if **save_to** is `None`.

        **batch_size** (`int`; default is `64`): the number of sentences per
        batch.

        **split** (`int`; default is `None`): the number of lines in sentences
        split. Allows to process a large dataset in pieces ("splits"). If
        **split** is `None` (default), all the dataset is processed without
        splits.

        **clone_ds** (`bool`; default is `False`): if `True`, the dataset is
        cloned and transformed. If `False`, `transform_collate` is used
        without cloning the dataset. There is no big differences between the
        variants. Both should produce identical results.

        **save_to** (`str`; default is `None`): the file name where the
        predictions will be saved.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

        Returns the corpus with feature keys and values predicted in the FEATS
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
                 use_cdict_coef=False, batch_size=BATCH_SIZE, split=None,
                 clone_ds=False, log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: the corpus of sentences with actual target values to score
        the tagger on. May be either the name of the file in *CoNLL-U* format
        or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

        **test** (default is `None`): the corpus of sentences with predicted
        target values. If `None` (default), the **gold** corpus will be
        retagged on-the-fly, and the result will be used as the **test**.

        **feats** (`str | list([str])`; default is `None`): one or several
        subfields of the key-value type fields like `FEATS` or `MISC` to be
        evaluated separatedly.

        **label** (`str`; default is `None`): the specific label of the target
        field to be evaluated separatedly, e.g. `field='UPOS', label='VERB'`
        or `field='FEATS:Animacy', label='Inan'`.

        **use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
        we use our prediction only. If `True`, we replace our prediction to
        the value returned by the `corpuscula.CorpusDict.predict_<field>()`
        method if its `coef` >= `.99`. Also, you can specify your own
        threshold as the value of the param.

        **batch_size** (`int`; default is `64`): the number of sentences per
        batch.

        **split** (`int`; default is `None`): the number of lines in sentences
        split. Allows to process a large dataset in pieces ("splits"). If
        **split** is `None` (default), all the dataset is processed without
        splits.

        **clone_ds** (`bool`; default is `False`): if `True`, the dataset is
        cloned and transformed. If `False`, `transform_collate` is used
        without cloning the dataset. There is no big differences between the
        variants. Both should produce identical results.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

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
              device=None, control_metric='accuracy', max_epochs=None,
              min_epochs=0, bad_epochs=5, batch_size=TRAIN_BATCH_SIZE,
              max_grad_norm=None, tags_to_remove=None, word_emb_type='bert',
              word_emb_path='xlm-roberta-base', word_transform_kwargs=None,
                  # BertDataset.transform() (for BERT-descendant models)
                  # params:
                  # {'max_len': 0, 'batch_size': 64, 'hidden_ids': '10',
                  #  'aggregate_hiddens_op': 'cat',
                  #  'aggregate_subtokens_op': 'absmax', 'to': junky.CPU,
                  #  'loglevel': 1}
                  # WordDataset.transform() (for other models) params:
                  # {'check_lower': True}
              stage1_params=None,
                  # {'lr': .0001, 'betas': (0.9, 0.999), 'eps': 1e-8,
                  #  'weight_decay': 0, 'amsgrad': False,
                  #  'max_epochs': None, 'min_epochs': None,
                  #  'bad_epochs': None, 'batch_size': None,
                  #  'max_grad_norm': None}
              stage2_params=None,
                  # {'lr': .001, 'momentum': .9, 'weight_decay': 0,
                  #  'dampening': 0, 'nesterov': False,
                  #  'max_epochs': None, 'min_epochs': None,
                  #  'bad_epochs': None, 'batch_size': None,
                  #  'max_grad_norm': None}
              stage3_params={'save_as': None},
                  # {'save_as': None, 'epochs': 3, 'batch_size': 8,
                  #  'lr': 2e-5, 'betas': (0.9, 0.999), 'eps': 1e-8,
                  #  'weight_decay': .01, 'amsgrad': False,
                  #  'num_warmup_steps': 3, 'max_grad_norm': 1.}
              stages=[1, 2, 3, 1, 2], save_stages=False, load_from=None,
              learn_on_padding=True, remove_padding_intent=False,
              seed=None, start_time=None, keep_embs=False, log_file=LOG_FILE,
              rnn_emb_dim=None, cnn_emb_dim=200, cnn_kernels=range(1, 7),
              upos_emb_dim=200, emb_bn=True, emb_do=.2,
              final_emb_dim=512, pre_bn=True, pre_do=.5,
              lstm_layers=1, lstm_do=0, tran_layers=0, tran_heads=8,
              post_bn=True, post_do=.4):
        """Creates and trains the key-value type field tagger model.

        During training, the best model is saved after each successful epoch.

        *Training's args*:

        **save_as** (`str`): the name using for save the model's head. Refer
        to the `.save()` method's help for the broad definition (see the
        **name** arg there).

        **device** (`str`, default is `None`): the device for the model. E.g.:
        'cuda:0'. If `None`, we don't move the model to any device (it is
        placed right where it's created).

        **control_metric** (`str`; default is `accuracy`): the metric that
        control training. Any that is supported by the `junky.train()` method.
        In the moment, it is: 'accuracy', 'f1', 'loss', 'precision', and
        'recall'.

        **max_epochs** (`int`; default is `None`): the maximal number of
        epochs for the model's head training (stages types `1` and `2`). If
        `None` (default), the training would be linger until **bad_epochs**
        has met, but no less than **min_epochs**.

        **min_epochs** (`int`; default is `0`): the minimal number of training
        epochs for the model's head training (stages types `1` and `2`).

        **bad_epochs** (`int`; default is `5`): the maximal allowed number of
        bad epochs (epochs when chosen **control_metric** is not became
        better) in a row for the model's head training (stages types `1` and
        `2`).

        **batch_size** (`int`; default is `32`): the number of sentences per
        batch for the model's head training (stages types `1` and `2`).

        **max_grad_norm** (`float`; default is `None`): the gradient clipping
        parameter for the model's head training (stages types `1` and `2`).

        **tags_to_remove** (`dict({str: str}) | dict({str: list([str])})`;
        default is `None`): the tags, tokens with those must be removed from
        the corpus. It's the `dict` with field names as keys and values you
        want to remove. Applied only to fields with atomic values (like
        *UPOS*). This argument may be used, for example, to remove some
        infrequent or just excess tags from the corpus. Note, that we remove
        the tokens from the train corpus completely, not just replace those
        tags to `None`.

        *Word embedding params*:

        **word_emb_type** (`str`; default is `'bert'`): one of (`'bert'` |
        `'glove'` | `'ft'` | `'w2v'`) embedding types.

        **word_emb_path** (`str`): the path to the word embeddings storage.

        **word_transform_kwargs** (`dict`; default is `None`): keyword
        arguments for the `.transform()` method of the dataset created for
        sentences to word embeddings conversion. See the `.transform()` method
        of either `junky.datasets.BertDataset` (if **word_emb_path** is
        `'bert'`) or `junky.datasets.WordDataset` (otherwise) if you want to
        learn allowed values for the parameter. If `None`, the `.transform()`
        method use its defaults.

        *Training stages params*:

        **stage1_param** (`dict`; default is `None`): keyword arguments for
        the `BaseModel.adjust_model_for_train()` method. If `None`, the
        defaults are used. Also, you can specify here new values for the
        arguments **max_epochs**, **min_epochs**, **bad_epochs**,
        **batch_size**, and **max_grad_norm** that will be used only on stages
        of type `1`.

        **stage2_param** (`dict`; default is `None`): keyword arguments for
        the `BaseModel.adjust_model_for_tune()` method. If `None`, the
        defaults are used. Also, you can specify here new values for the
        arguments **max_epochs**, **min_epochs**, **bad_epochs**,
        **batch_size**, and **max_grad_norm** that will be used only on stages
        of type `2`.

        **stage3_param** (`dict`; default is `None`): keyword arguments for
        the `WordEmbeddings.full_tune()` method. If `None`, the defaults are
        used.

        **stages** (`list([int]`; default is `[1, 2, 3, 1, 2]`): what stages
        we should use during training and in which order. On the stage type
        `1` the model head is trained with *Adam* optimizer; the stage type
        `2` is similar, but the optimizer is *SGD*; the stage type `3` is only
        relevant when **word_emb_type** is `'bert'` and we want to tune the
        whole model. Stage type `0` defines the skip-stage, i.e. there would
        be no real training on it. It is used when you need reproducibility
        and want to continue train the model from some particular stage. In
        this case, you specify the name of the model saved on that stage in
        the parametere **load_from**, and put zeros into the **stages** list
        on the places of already finished ones. One more time: it is used for
        reproducibility only, i.e. when you put some particular value to the
        **seed** param and want the data order in bathes be equivalent with
        data on the stages from the past trainings.

        **save_stages** (`bool`; default is `False`): if we need to keep the
        best model of each stage beside of the overall best model. The names
        of these models would have the suffix `_<idx>(stage<stage_type>)`
        where `<idx>` is an ordinal number of the stage. We can then use it to
        continue training from any particular stage number (changing next
        stages or their parameters) using the parameter **load_from**. Note
        that we save only stages of the head model. The embedding model as a
        part of the full model usually tune only once, so we don't make its
        copy.

        **load_from** (`str`; default is `None`): if you want to continue
        training from one of previously saved stages, you can specify the name
        of the model from that stage. Note, that if your model is already
        trained on stage type `3`, then you want to set param
        **word_emb_path** to `None`. Otherwise, you'll load wrong embedding
        model. Any other params of the model may be overwritten (and most
        likely, this would cause error), but they are equivalent when the
        training is just starts and when it's continues. But the
        **word_emb_path** is different if you already passed stage type `3`,
        so don't forget to set it to `None` in that case. (Example: you want
        to repeat training on stage no `5`, so you specify in the
        **load_from** param something like `'model_4(stage1)'` and set the
        **word_emb_path** to `None` and the **stages_param** to
        `'[0, 0, 0, 0, 2]'` (or, if you don't care of reproducibility, you
        could just specify `[2]` here).

        *Other options*:

        **learn_on_padding** (`bool`; default is `True`): while training, we
        can calculate loss either taking in account predictions made for
        padding tokens or without it. The common practice is don't use padding
        when calculate loss. However, we note that using padding sometimes
        makes the resulting model performance slightly better.

        **remove_padding_intent** (`bool`; default is `False`): if you set
        **learn_on_padding** param to `False`, you may want not to use padding
        intent during training at all. I.e. padding tokens would be tagged
        with some of real tags, and they would just ignored during computing
        loss. As a result, the model would have the output dimensionality of
        the final layer less by one. On the first sight, such approach could
        increase the performance, but in our experiments, such effect appeared
        not always.

        **seed** (`int`; default is `None`): init value for the random number
        generator if you need reproducibility. Note that each stage will have
        its own seed value, and the **seed** param is used to calculate these
        values.

        **start_time** (`float`; default is `None`): the result of
        `time.time()` to start with. If `None`, the arg will be init anew.

        **keep_embs** (`bool`; default is `False`): by default, after creating
        `Dataset` objects, we remove word embedding models to free memory.
        With `keep_embs=False` this operation is omitted, and you can use
        `.embs` attribute for share embedding models with other objects.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

        *The model hyperparameters*:

        **rnn_emb_dim** (`int`; default is `None`): the internal character RNN
        (LSTM) embedding dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`; default is `200`): the internal character CNN
        embedding dimensionality. If `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
        kernel sizes of the internal CNN embedding layer. Relevant if
        **cnn_emb_dim** is not `None`.

        **upos_emb_dim** (`int`; default is `200`): the auxiliary UPOS label
        embedding dimensionality.

        **emb_bn** (`bool`; default is `True`): whether batch normalization
        layer should be applied after the embedding concatenation.

        **emb_do** (`float`; default is `.2`): the dropout rate after the
        embedding concatenation.

        **final_emb_dim** (`int`; default is `512`): the output dimesionality
        of the linear transformation applying to concatenated embeddings.

        **pre_bn** (`bool`; default is `True`): whether batch normalization
        layer should be applied before the main part of the algorithm.

        **pre_do** (`float`; default is `.5`): the dropout rate before the
        main part of the algorithm.

        **lstm_layers** (`int`; default is `1`): the number of Bidirectional
        LSTM layers. If `None`, they are not created.

        **lstm_do** (`float`; default is `0`): the dropout between LSTM
        layers. Only relevant, if `lstm_layers` > `1`.

        **tran_layers** (`int`; default is `None`): the number of Transformer
        Encoder layers. If `None`, they are not created.

        **tran_heads** (`int`; default is `8`): the number of attention heads
        of Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

        **post_bn** (`bool`; default is `True`): whether batch normalization
        layer should be applied after the main part of the algorithm.

        **post_do** (`float`; default is `.4`): the dropout rate after the
        main part of the algorithm.

        The method returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if not start_time:
            start_time = time.time()
        args, kwargs = get_func_params(FeatsJointTagger.train, locals())

        for _ in (
            x.update({self._field: '|'.join('='.join((y, x[self._field][y]))
                                       for y in sorted(x[self._field]))})
                for x in [self._train_corpus,
                          self._test_corpus if self._test_corpus else []]
                for x in x
                for x in x
        ):
            pass

        if self._test_corpus:
            key_vals = set(x[self._field] for x in self._train_corpus
                                          for x in x)
            for _ in (
                None if x[self._field] in key_vals else
                x.update({self._field: [*get_close_matches(x[self._field],
                                                           key_vals, n=1),
                                        ''][0]})
                    for x in self._test_corpus for x in x
            ):
                pass

        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)


class FeatsSeparateTagger(BaseTagger):
    """
    A class for the prediction of the content ot a key-value type field.
    Separate implementation (for each feature, creates its own tagger).

    Args:

    **field** (`str`; default is `FEATS`): the name of the *CoNLL-U* key-value
    type field, content of which needs to be predicted. With this tagger, you
    can predict only key-value type fields, like FEATS.

    **embs** (`dict({str: object}); default is `None`): the `dict` with paths
    to embeddings files as keys and corresponding embedding models as values.
    If the tagger needs to load any embedding model, firstly, the model is
    looked up it in that `dict`.

    During init, **embs** is copied to the `embs` attribute of the creating
    object, and this attribute may be used further to share already loaded
    embeddings with another taggers.
    """
    def __init__(self, field='FEATS', feats_prune_coef=6, embs=None):
        super().__init__(embs=embs)
        self._field = field
        self._feats_prune_coef = feats_prune_coef
        self._feats = {}

    def save(self, name, log_file=LOG_FILE):
        """Saves the config of tagger with paths to the separate taggers'
        storages.

        Args:

        **name**: the name to save with.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
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

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
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
            tagger = FeatTagger(self._field + ':' + feat,
                                feats_prune_coef=self._feats_prune_coef,
                                embs=self.embs)
            tagger.load(name_)
            self._feats[feat] = [name, tagger]
        if log_file:
            print('\n{} tagger have been loaded ###'.format(self._field),
                  file=log_file)

    def predict(self, corpus, feats=None, remove_excess_feats=True,
                use_cdict_coef=False, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts feature keys and values in the FEATS field of the corpus.

        Args:

        **corpus**: the corpus which will be used for the feature extraction
        and predictions. May be either the name of the file in *CoNLL-U*
        format or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

        **feats** (`str | list([str])`; default is `None`): one or several
        subfields of the key-value type fields like `FEATS` or `MISC` to be
        predicted separatedly.

        **remove_excess_feats** (`bool`): if `True` (default), the tagger
        removes all unrelevant features from the predicted field ("unrelevant"
        means, that the tagger don't have a models for them). For example, if
        you trained the tagger only for "Case" and "Gender" features, the
        tagger predicts only them (or, only one of them, if you specify it in
        the **feats** field) and removes all the rest. Elsewise, if
        **remove_excess_feats** is `False`, all unrelevant feats are stayed
        intact.

        **use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
        we use our prediction only. If `True`, we replace our prediction to
        the value returned by the `corpuscula.CorpusDict.predict_<field>()`
        method if its `coef` >= `.99`. Also, you can specify your own
        threshold as the value of the param.

        **with_orig** (`bool`; default is `False`): if `True`, instead of just
        the sequence with predicted labels, return the sequence of tuples
        where the first element is the sentence with predicted labels and the
        second element is the original sentence. **with_orig** can be `True`
        only if **save_to** is `None`.

        **batch_size** (`int`; default is `64`): the number of sentences per
        batch.

        **split** (`int`; default is `None`): the number of lines in sentences
        split. Allows to process a large dataset in pieces ("splits"). If
        **split** is `None` (default), all the dataset is processed without
        splits.

        **clone_ds** (`bool`; default is `False`): if `True`, the dataset is
        cloned and transformed. If `False`, `transform_collate` is used
        without cloning the dataset. There is no big differences between the
        variants. Both should produce identical results.

        **save_to** (`str`; default is `None`): the file name where the
        predictions will be saved.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

        Returns the corpus with feature keys and values predicted in the FEATS
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
                 use_cdict_coef=False, batch_size=BATCH_SIZE, split=None,
                 clone_ds=False, log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: the corpus of sentences with actual target values to score
        the tagger on. May be either the name of the file in *CoNLL-U* format
        or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

        **test** (default is `None`): the corpus of sentences with predicted
        target values. If `None` (default), the **gold** corpus will be
        retagged on-the-fly, and the result will be used as the **test**.

        **feats** (`str | list([str])`; default is `None`): one or several
        subfields of the key-value type fields like `FEATS` or `MISC` to be
        evaluated separatedly.

        **label** (`str`; default is `None`): the specific label of the target
        field to be evaluated separatedly, e.g. `field='UPOS', label='VERB'`
        or `field='FEATS:Animacy', label='Inan'`.

        **use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
        we use our prediction only. If `True`, we replace our prediction to
        the value returned by the `corpuscula.CorpusDict.predict_<field>()`
        method if its `coef` >= `.99`. Also, you can specify your own
        threshold as the value of the param.

        **batch_size** (`int`; default is `64`): the number of sentences per
        batch.

        **split** (`int`; default is `None`): the number of lines in sentences
        split. Allows to process a large dataset in pieces ("splits"). If
        **split** is `None` (default), all the dataset is processed without
        splits.

        **clone_ds** (`bool`; default is `False`): if `True`, the dataset is
        cloned and transformed. If `False`, `transform_collate` is used
        without cloning the dataset. There is no big differences between the
        variants. Both should produce identical results.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

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
              device=None, control_metric='accuracy', max_epochs=None,
              min_epochs=0, bad_epochs=5, batch_size=TRAIN_BATCH_SIZE,
              max_grad_norm=None, tags_to_remove=None, word_emb_type='bert',
              word_emb_path=None, word_transform_kwargs=None,
                  # BertDataset.transform() (for BERT-descendant models)
                  # params:
                  # {'max_len': 0, 'batch_size': 64, 'hidden_ids': '10',
                  #  'aggregate_hiddens_op': 'cat',
                  #  'aggregate_subtokens_op': 'absmax', 'to': junky.CPU,
                  #  'loglevel': 1}
                  # WordDataset.transform() (for other models) params:
                  # {'check_lower': True}
              stage1_params=None,
                  # {'lr': .0001, 'betas': (0.9, 0.999), 'eps': 1e-8,
                  #  'weight_decay': 0, 'amsgrad': False,
                  #  'max_epochs': None, 'min_epochs': None,
                  #  'bad_epochs': None, 'batch_size': None,
                  #  'max_grad_norm': None}
              stage2_params=None,
                  # {'lr': .001, 'momentum': .9, 'weight_decay': 0,
                  #  'dampening': 0, 'nesterov': False,
                  #  'max_epochs': None, 'min_epochs': None,
                  #  'bad_epochs': None, 'batch_size': None,
                  #  'max_grad_norm': None}
              stage3_params={'save_as': None},
                  # {'save_as': None, 'epochs': 3, 'batch_size': 8,
                  #  'lr': 2e-5, 'betas': (0.9, 0.999), 'eps': 1e-8,
                  #  'weight_decay': .01, 'amsgrad': False,
                  #  'num_warmup_steps': 3, 'max_grad_norm': 1.}
              stages=[1, 2, 3, 1, 2], save_stages=False,
              learn_on_padding=True, remove_padding_intent=False,
              seed=None, keep_embs=False, log_file=LOG_FILE,
              rnn_emb_dim=None, cnn_emb_dim=200, cnn_kernels=range(1, 7),
              upos_emb_dim=200, emb_bn=True, emb_do=.2,
              final_emb_dim=512, pre_bn=True, pre_do=.5,
              lstm_layers=1, lstm_do=0, tran_layers=0, tran_heads=8,
              post_bn=True, post_do=.4):
        """Creates and trains the separate feature tagger model.

        *Training's args*:

        **save_as** (`str`): the name using for save the model's head. Refer
        to the `.save()` method's help for the broad definition (see the
        **name** arg there).

        **feats** (`str | list([str])`; default is `None`): one or several
        subfields of the key-value type fields like `FEATS` or `MISC` to be
        predicted separatedly.

        **device** (`str`; default is `None`): the device for the model. E.g.:
        'cuda:0'. If `None`, we don't move the model to any device (it is
        placed right where it's created).

        **control_metric** (`str`; default is `accuracy`): the metric that
        control training. Any that is supported by the `junky.train()` method.
        In the moment, it is: 'accuracy', 'f1', 'loss', 'precision', and
        'recall'.

        **max_epochs** (`int`; default is `None`): the maximal number of
        epochs for the model's head training (stages types `1` and `2`). If
        `None` (default), the training would be linger until **bad_epochs**
        has met, but no less than **min_epochs**.

        **min_epochs** (`int`; default is `0`): the minimal number of training
        epochs for the model's head training (stages types `1` and `2`).

        **bad_epochs** (`int`; default is `5`): the maximal allowed number of
        bad epochs (epochs when chosen **control_metric** is not became
        better) in a row for the model's head training (stages types `1` and
        `2`).

        **batch_size** (`int`; default is `32`): the number of sentences per
        batch for the model's head training (stages types `1` and `2`).

        **max_grad_norm** (`float`; default is `None`): the gradient clipping
        parameter for the model's head training (stages types `1` and `2`).

        **tags_to_remove** (`dict({str: str}) | dict({str: list([str])})`;
        default is `None`): the tags, tokens with those must be removed from
        the corpus. It's the `dict` with field names as keys and values you
        want to remove. Applied only to fields with atomic values (like
        *UPOS*). This argument may be used, for example, to remove some
        infrequent or just excess tags from the corpus. Note, that we remove
        the tokens from the train corpus completely, not just replace those
        tags to `None`.

        *Word embedding params*:

        **word_emb_type** (`str`; default is `'bert'`): one of (`'bert'` |
        `'glove'` | `'ft'` | `'w2v'`) embedding types.

        **word_emb_path** (`str`): the path to the word embeddings storage.

        **word_transform_kwargs** (`dict`; default is `None`): keyword
        arguments for the `.transform()` method of the dataset created for
        sentences to word embeddings conversion. See the `.transform()` method
        of either `junky.datasets.BertDataset` (if **word_emb_path** is
        `'bert'`) or `junky.datasets.WordDataset` (otherwise) if you want to
        learn allowed values for the parameter. If `None`, the `.transform()`
        method use its defaults.

        *Training stages params*:

        **stage1_param** (`dict`; default is `None`): keyword arguments for
        the `BaseModel.adjust_model_for_train()` method. If `None`, the
        defaults are used. Also, you can specify here new values for the
        arguments **max_epochs**, **min_epochs**, **bad_epochs**,
        **batch_size**, and **max_grad_norm** that will be used only on stages
        of type `1`.

        **stage2_param** (`dict`; default is `None`): keyword arguments for
        the `BaseModel.adjust_model_for_tune()` method. If `None`, the
        defaults are used. Also, you can specify here new values for the
        arguments **max_epochs**, **min_epochs**, **bad_epochs**,
        **batch_size**, and **max_grad_norm** that will be used only on stages
        of type `2`.

        **stage3_param** (`dict`; default is `None`): keyword arguments for
        the `WordEmbeddings.full_tune()` method. If `None`, the defaults are
        used.

        **stages** (`list([int]`; default is `[1, 2, 3, 1, 2]`): what stages
        we should use during training and in which order. On the stage type
        `1` the model head is trained with *Adam* optimizer; the stage type
        `2` is similar, but the optimizer is *SGD*; the stage type `3` is only
        relevant when **word_emb_type** is `'bert'` and we want to tune the
        whole model. Stage type `0` defines the skip-stage, i.e. there would
        be no real training on it. It is used when you need reproducibility
        and want to continue train the model from some particular stage. In
        this case, you specify the name of the model saved on that stage in
        the parametere **load_from**, and put zeros into the **stages** list
        on the places of already finished ones. One more time: it is used for
        reproducibility only, i.e. when you put some particular value to the
        **seed** param and want the data order in bathes be equivalent with
        data on the stages from the past trainings.

        **save_stages** (`bool`; default is `False`): if we need to keep the
        best model of each stage beside of the overall best model. The names
        of these models would have the suffix `_<idx>(stage<stage_type>)`
        where `<idx>` is an ordinal number of the stage. We can then use it to
        continue training from any particular stage number (changing next
        stages or their parameters) using the parameter **load_from** (with
        separate tagger the tuning is more complicated: one have to load and
        tune each model separately as standalone instance of the `FeatTagger`
        class; the `FeatsSeparateTagger.predict()` method doesn't even have
        **load_from** argument). Note that we save only stages of the head
        model. The embedding model as a part of the full model usually tune
        only once, so we don't make its copy.

        *Other options*:

        **learn_on_padding** (`bool`; default is `True`): while training, we
        can calculate loss either taking in account predictions made for
        padding tokens or without it. The common practice is don't use padding
        when calculate loss. However, we note that using padding sometimes
        makes the resulting model performance slightly better.

        **remove_padding_intent** (`bool`; default is `False`): if you set
        **learn_on_padding** param to `False`, you may want not to use padding
        intent during training at all. I.e. padding tokens would be tagged
        with some of real tags, and they would just ignored during computing
        loss. As a result, the model would have the output dimensionality of
        the final layer less by one. On the first sight, such approach could
        increase the performance, but in our experiments, such effect appeared
        not always.

        **seed** (`int`; default is `None`): init value for the random number
        generator if you need reproducibility. Note that each stage will have
        its own seed value, and the **seed** param is used to calculate these
        values.

        **keep_embs** (`bool`; default is `False`): by default, after creating
        `Dataset` objects, we remove word embedding models to free memory.
        With `keep_embs=False` this operation is omitted, and you can use
        `.embs` attribute for share embedding models with other objects.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.

        *The model hyperparameters*:

        **rnn_emb_dim** (`int`; default is `None`): the internal character RNN
        (LSTM) embedding dimensionality. If `None`, the layer is skipped.

        **cnn_emb_dim** (`int`; default is `200`): the internal character CNN
        embedding dimensionality. If `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
        kernel sizes of the internal CNN embedding layer. Relevant if
        **cnn_emb_dim** is not `None`.

        **upos_emb_dim** (`int`; default is `200`): the auxiliary UPOS label
        embedding dimensionality.

        **emb_bn** (`bool`; default is 'True'): whether batch normalization
        layer should be applied after the embedding concatenation.

        **emb_do** (`float`; default is '.2'): the dropout rate after the
        embedding concatenation.

        **final_emb_dim** (`int`; default is `512`): the output dimesionality
        of the linear transformation applying to concatenated embeddings.

        **pre_bn** (`bool`; default is 'True'): whether batch normalization
        layer should be applied before the main part of the algorithm.

        **pre_do** (`float`; default is '.5'): the dropout rate before the
        main part of the algorithm.

        **lstm_layers** (`int`; default is `1`): the number of Bidirectional
        LSTM layers. If `None`, they are not created.

        **lstm_do** (`float`; default is `0`): the dropout between LSTM
        layers. Only relevant, if `lstm_layers` > `1`.

        **tran_layers** (`int`; default is `None`): the number of Transformer
        Encoder layers. If `None`, they are not created.

        **tran_heads** (`int`; default is `8`): the number of attention heads
        of Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

        **post_bn** (`bool`; default is 'True'): whether batch normalization
        layer should be applied after the main part of the algorithm.

        **post_do** (`float`; default is '.4'): the dropout rate after the
        main part of the algorithm.

        The method returns the train statistics.
        """
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        start_time = time.time()
        args, kwargs = get_func_params(FeatsSeparateTagger.train, locals())
        del kwargs['feats']
        if 'stage3_params' in kwargs and 'save_as' in kwargs['stage3_params']:
            word_emb_path_suffix = kwargs['stage3_params']['save_as']
            del kwargs['stage3_params']['save_as']
        else:
            word_emb_path_suffix = None

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
        elif isinstance(feats, str):
            feats = [feats]
        if log_file:
            print(', '.join(feats), file=log_file)

        res = {}
        for feat in feats:
            start_time_ = time.time()
            if log_file:
                print(file=log_file)
                clear_tqdm()

            save_as_ = '{}-{}'.format(save_as, feat.lower())
            self._feats[feat] = save_as_

            tagger = FeatTagger(self._field + ':' + feat,
                                feats_prune_coef=self._feats_prune_coef,
                                embs=self.embs)
            tagger._train_corpus, tagger._test_corpus = \
                self._train_corpus, self._test_corpus
            if word_emb_path_suffix:
                kwargs['stage3_params']['save_as'] = \
                    '{}-{}_{}'.format(self._field.lower(), feat.lower(),
                                      word_emb_path_suffix)
            res[feat] = tagger.train(save_as_, **kwargs,
                                     start_time=start_time_)
            del tagger

        self.save(save_as, log_file=log_file)
        if log_file:
            print('\n###### {} TAGGER TRAINING HAS FINISHED ### '
                      .format(self._field)
                + 'Total time: {} ######\n'
                      .format(seconds_to_strtime(time.time() - start_time)),
                  file=log_file)
            print(("Now, check the separate {} models' and datasets' "
                   'config files and consider to change some device names '
                   'to be able to load all the models jointly. You can find '
                   'the separate models\' list in the "{}" config file. '
                   "Then, use the `.load('{}')` method to start working "
                   'with the {} tagger.')
                       .format(self._field, save_as + CONFIG_EXT, save_as,
                               self._field),
                      file=log_file)
        return res
