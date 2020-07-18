# -*- coding: utf-8 -*-
# MorDL project: FEATS:feat tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a single-tag FEAT tagger class.
"""
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class FeatTagger(BaseTagger):
    """
    A class for single-tag FEAT feature tagging.
    """

    def __init__(self, feat):
        super().__init__()
        if feat.find(':') == -1:
            feat = 'FEATS:' + feat
        self._feat = feat

    def load(self, name, device=None, dataset_device=None, log_file=LOG_FILE):
        """Loads feature tagger and dataset.

        Args:

        **name** (`str`): name of the internal state previously saved.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        overrride its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        args, kwargs = get_func_params(FeatTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the specified FEAT fields for the corpus.

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

        Returns corpus with tag predictions in the FEAT field.
        """
        args, kwargs = get_func_params(FeatTagger.predict, locals())
        return super().predict(self._feat, 'UPOS', *args, **kwargs)

    def evaluate(self, gold, test=None, label=None, batch_size=BATCH_SIZE,
                 split=None, clone_ds=False, log_file=LOG_FILE):
        """Evaluate predicitons on the development test set.

        Args:

        **gold** (`tuple(<sentences> <labels>)`): corpus with actual target
        tags.

        **test** (`tuple(<sentences> <labels>)`): corpus with predicted target
        tags. If `None`, predictions will be created on-the-fly based on the
        `gold` corpus.

        **label** (`str`): specific label of the target field to be evaluated,
        e.g. `field='UPOS'`, `label='VERB'` or `field='FEATS:Animacy'`,
        `label='Inan'`. Note that to evaluate key-value type fields like
        `FEATS` or `MISC`

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
        args, kwargs = get_func_params(FeatTagger.evaluate, locals())
        return super().evaluate(self._feat, *args, **kwargs)

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
        args, kwargs = get_func_params(FeatTagger.train, locals())
        return super().train(self._feat, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)
