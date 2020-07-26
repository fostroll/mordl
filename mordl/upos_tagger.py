# -*- coding: utf-8 -*-
# MorDL project: UPOS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a UPOS tagger class.
"""
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.upos_tagger_model import UposTaggerModel


class UposTagger(BaseTagger):
    """
    A UPOS tagger class.

    Args:

    **field**: a name of the *CoNLL-U* field, values of which needs to be
    predicted. With this tagger, you can predict only fields with atomic
    values, like UPOS.
    """
    def __init__(self, field='UPOS'):
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
        args, kwargs = get_func_params(UposTagger.load, locals())
        super().load(UposTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the UPOS field of the corpus.

        Args:

        **corpus**: a corpus which will be used for feature extraction and
        predictions. May be either a name of the file in *CoNLL-U* format or a
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

        Returns corpus with tags predicted in the UPOS field.
        """
        args, kwargs = get_func_params(UposTagger.predict, locals())
        return super().predict(self._field, None, *args, **kwargs)

    def evaluate(self, gold, test=None, label=None, batch_size=BATCH_SIZE,
                 split=None, clone_ds=False, log_file=LOG_FILE):
        """Evaluate the tagger model.

        Args:

        **gold**: a corpus of sentences with actual target values to score the
        tagger on. May be either a name of the file in *CoNLL-U* format or a
        list/iterator of sentences in *Parsed CoNLL-U*.

        **test**: a corpus of sentences with predicted target values. If
        `None`, the **gold** corpus will be retagged on-the-fly, and the
        result will be used as **test**.

        **label** (`str`): specific label of the target field to be evaluated,
        e.g. label='VERB'`.

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
        args, kwargs = get_func_params(UposTagger.evaluate, locals())
        return super().evaluate(self._field, *args, **kwargs)

    def train(self, save_as,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=3, lstm_do=0,
              bn1=True, do1=.2, bn2=True, do2=.5, bn3=True, do3=.4, seed=None,
              log_file=LOG_FILE):
        """Creates and trains the UPOS tagger model.

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
        args, kwargs = get_func_params(UposTagger.train, locals())
        return super().train(self._field, None, UposTaggerModel, None,
                             *args, **kwargs)
