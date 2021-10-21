# -*- coding: utf-8 -*-
# MorDL project: UPOS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a UPOS tagger class.
"""
from corpuscula import CorpusDict
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, CDICT_COEF_THRESH ,LOG_FILE, \
                           TRAIN_BATCH_SIZE
from mordl.upos_tagger_model import UposTaggerModel
import time

_CDICT_COEF_THRESH = .99


class UposTagger(BaseTagger):
    """
    The UPOS tagger class.

    Args:

    **field** (`str`; default is `UPOS`): the name of the *CoNLL-U* field,
    values of which needs to be predicted. With this tagger, you can predict
    only fields with atomicvalues, like UPOS.

    **embs** (`dict({str: object}); default is `None`): the `dict` with paths
    to embeddings files as keys and corresponding embedding models as values.
    If the tagger needs to load any embedding model, firstly, the model is
    looked up it in that `dict`.

    During init, **embs** is copied to the `embs` attribute of the creating
    object, and this attribute may be used further to share already loaded
    embeddings with another taggers.
    """
    def __init__(self, field='UPOS', embs=None):
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
        args, kwargs = get_func_params(UposTagger.load, locals())
        super().load(UposTaggerModel, *args, **kwargs)

    def _check_cdict(self, sentence, use_cdict_coef):
        if use_cdict_coef not in [None, False]:
            isfirst = True
            for token in sentence[0] if isinstance(sentence, tuple) else \
                sentence:
                id_, form = token['ID'], token['FORM']
                if form and '-' not in id_:
                    guess, coef = \
                        self._cdict.predict_tag(form, isfirst=isfirst)
                    isfirst = False
                    if coef is not None \
                   and coef >= (CDICT_COEF_THRESH
                                    if use_cdict_coef is True else
                                use_cdict_coef):
                        token['UPOS'] = guess
        return sentence

    def predict(self, corpus, use_cdict_coef=False, with_orig=False,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                save_to=None, log_file=LOG_FILE):
        """Predicts tags in the UPOS field of the corpus.

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

        Returns the corpus with tags predicted in the UPOS field.
        """
        assert self._field == 'UPOS' or use_cdict_coef in [None, False], \
            'ERROR: "use_cdict_coef" param may be used only with UPOS field'
        args, kwargs = get_func_params(UposTagger.predict, locals())
        return super().predict(self._field, None, *args, **kwargs)

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
        args, kwargs = get_func_params(UposTagger.evaluate, locals())
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
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              emb_bn=True, emb_do=.2,
              final_emb_dim=512, pre_bn=True, pre_do=.5,
              lstm_layers=1, lstm_do=0, tran_layers=0, tran_heads=8,
              post_bn=True, post_do=.4):
        """Creates and trains the UPOS tagger model.

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
        UPOS). This argument may be used, for example, to remove some
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

        **cnn_emb_dim** (`int`; default is `None`): the internal character CNN
        embedding dimensionality. If `None`, the layer is skipped.

        **cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
        kernel sizes of the internal CNN embedding layer. Relevant if
        **cnn_emb_dim** is not `None`.

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
        args, kwargs = get_func_params(UposTagger.train, locals())

        self._cdict = CorpusDict(
            corpus=(x for x in [self._train_corpus,
                                self._test_corpus if self._test_corpus else
                                []]
                      for x in x),
            format='conllu_parsed', log_file=log_file
        )

        return super().train(self._field, None, UposTaggerModel, None,
                             *args, **kwargs)
