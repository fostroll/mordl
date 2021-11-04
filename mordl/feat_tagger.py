# -*- coding: utf-8 -*-
# MorDL project: FEATS:feat tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a single-tag FEAT tagger class.
"""
from corpuscula import CorpusDict
from difflib import get_close_matches
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel
import time


class FeatTagger(BaseTagger):
    """
    The class of single-feature tagger.

    Args:

    **feat** (`str`): the name of the feat which needs to be predicted by the
    tagger. May contains a prefix, separated by the colon (`:`). In that case,
    the prefix is treated as a field name. Also, if **feat** starts with the
    colon, we get `'FEATS'` as a field name. Otherwise, if **feat** contains
    no colon, the field name set to **feat** itself. Examples: `'MISC:NE'`,
    `':Animacy'`, `'DEPREL'`.

    **feats_prune_coef** (`int`; default is `6`): the feature prunning
    coefficient which allows to eliminate all features that have a low
    frequency. To improve the prediction quality, we get a number of
    occurences of the most frequent feature from the FEATS field for each UPOS
    tag, divide that number by **feats_prune_coef**, and use only those
    features, the number of occurences of which is greater than that value.
    * `feats_prune_coef=0` means "do not use feats";
    * `feats_prune_coef=None` means "use all feats";
    * default `feats_prune_coef=6`.

    **NB**: the argument is relevant only if **feat** is not from FEATS field.

    **embs** (`dict({str: object}); default is `None`): the `dict` with paths
    to embeddings files as keys and corresponding embedding models as values.
    If the tagger needs to load any embedding model, firstly, the model is
    looked up it in that `dict`.

    During init, **embs** is copied to the `embs` attribute of the creating
    object, and this attribute may be used further to share already loaded
    embeddings with another taggers.
    """
    def __init__(self, feat, feats_prune_coef=6, embs=None):
        assert feat != 'UPOS', 'ERROR: for UPOS field use UposTagger class'
        super().__init__(embs=embs)
        if feat[0] == ':':
            feat = 'FEATS' + feat
        self._field = feat
        if feat.startswith('FEATS:'):
            feats_prune_coef = 0
        self._feats_prune_coef = feats_prune_coef
        self._model_class = FeatTaggerModel

    def _transform_upos(self, corpus, key_vals=None):
        rel_feats = {}
        prune_coef = self._feats_prune_coef
        if prune_coef != 0:
            tags = self._cdict.get_tags_freq()
            if tags:
                thresh = tags[0][1] / prune_coef if prune_coef else 0
                tags = [x[0] for x in tags if x[1] > thresh]
            for tag in tags:
                feats_ = rel_feats[tag] = set()
                tag_feats = self._cdict.get_feats_freq(tag)
                if tag_feats:
                    thresh = tag_feats[0][1] / prune_coef if prune_coef else 0
                    [feats_.add(x[0]) for x in tag_feats if x[1] > thresh]

        for sent in corpus:
            for tok in sent[0] if isinstance(sent, tuple) else sent:
                upos = tok['UPOS']
                if upos:
                    upos = upos_ = upos.split(' ')[0]
                    feats = tok['FEATS']
                    if feats:
                        rel_feats_ = rel_feats.get(upos)
                        if rel_feats_:
                            for feat, val in sorted(feats.items()):
                                if feat in rel_feats_:
                                    upos_ += ' ' + feat + ':' + val
                        upos = upos_ \
                                   if not key_vals or upos_ in key_vals else \
                               [*get_close_matches(upos_,
                                                   key_vals, n=1), upos][0]
                    tok['UPOS'] = upos
            yield sent

    @staticmethod
    def _restore_upos(corpus, with_orig=False):
        def restore_upos(sent):
            if isinstance(sent, tuple):
                sent = sent[0]
            for tok in sent:
                upos = tok['UPOS']
                if upos:
                    tok['UPOS'] = upos.split(' ')[0]
        for sent in corpus:
            for tok in sent:
                if with_orig:
                    restore_upos(sent[0])
                    restore_upos(sent[1])
                else:
                    restore_upos(sent)
            yield sent

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
        args, kwargs = get_func_params(FeatTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, use_cdict_coef=False, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE, **_):
        """Predicts values in the certain feature of the key-value type field
        of the specified **corpus**.

        Args:

        **corpus**: a corpus which will be used for feature extraction and
        predictions. May be either a name of the file in *CoNLL-U* format or a
        list/iterator of sentences in *Parsed CoNLL-U*.

        **use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
        we use our prediction only. If `True`, we replace our prediction to
        the value returned by the `corpuscula.CorpusDict.predict_<field>()`
        method if its `coef` >= `.99`. Also, you can specify your own
        threshold as the value of the param.

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

        Returns the corpus with predicted values of certain feature in the
        key-value type field.
        """
        assert self._ds is not None, \
               "ERROR: The tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(FeatTagger.predict, locals())

        if self._feats_prune_coef != 0:
            kwargs['save_to'] = None
            key_vals = self._ds.get_dataset('t_0').transform_dict
            corpus = self._get_corpus(corpus, asis=True, log_file=log_file)
            corpus = self._transform_upos(corpus, key_vals=key_vals)

        corpus = super().predict(self._field, 'UPOS', corpus, **kwargs)

        if self._feats_prune_coef != 0:
            corpus = self._restore_upos(corpus)

            if save_to:
                self.save_conllu(corpus, save_to, log_file=None)
                corpus = self._get_corpus(save_to, asis=True, log_file=log_file)

        return corpus

    def evaluate(self, gold, test=None, label=None, use_cdict_coef=False,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: the corpus of sentences with actual target values to score
        the tagger on. May be either the name of the file in *CoNLL-U* format
        or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

        **test** (default is `None`): the corpus of sentences with predicted
        target values. If `None` (default), the **gold** corpus will be
        retagged on-the-fly, and the result will be used as the **test**.

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
        args, kwargs = get_func_params(FeatTagger.evaluate, locals())
        return super().evaluate(self._field, *args, **kwargs)

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
        """Creates and trains the feature tagger model.

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
        if not start_time:
            start_time = time.time()
        args, kwargs = get_func_params(FeatTagger.train, locals())

        if self._feats_prune_coef != 0:
            for _ in (
                x.update({'LEMMA': x['FORM']})
                    for x in [self._train_corpus,
                              self._test_corpus if self._test_corpus else []]
                    for x in x
                    for x in x
            ):
                pass
            self._cdict = CorpusDict(
                corpus=(x for x in [self._train_corpus,
                                    self._test_corpus
                                        if self._test_corpus else
                                    []]
                          for x in x),
                format='conllu_parsed', log_file=log_file
            )
            self._save_cdict(save_as + '.cdict.pickle')
            if log_file:
                print(file=log_file)

            if self._test_corpus:
                for _ in self._transform_upos(self._train_corpus):
                    pass
                key_vals = set(x['UPOS'] for x in self._train_corpus
                                         for x in x
                                   if x['FORM'] and x['UPOS']
                                                and '-' not in x['ID'])
                for _ in self._transform_upos(self._test_corpus, key_vals):
                    pass

        return super().train(self._field, 'UPOS', self._model_class, 'upos',
                             *args, **kwargs)
