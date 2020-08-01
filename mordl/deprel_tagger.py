# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a Named-entity tagger class.
"""
from copy import deepcopy
from junky import get_func_params
from mordl import FeatTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.deprel_tagger_model import DeprelTaggerModel

WINDOW_LEFT, WINDOW_RIGHT = 2, 2
PAD = '[PAD]'
PAD_TOKEN = {'ID': '0', 'FORM': PAD, 'LEMMA': PAD,
                        'UPOS': PAD, 'FEATS': {}, 'DEPREL': None}


class DeprelTagger(FeatTagger):
    """
    Named-entity tagger class. We use the feature 'NE' of MISC field as the
    place where Named-entity tags are stored.

    Args:

    **feats_prune_coef** (`int`): feature prunning coefficient which allows to
    eliminate all features that have a low frequency. For each UPOS tag, we
    get a number of occurences of the most frequent feature from FEATS field,
    divide it by **feats_prune_coef** and use only those features, number of
    occurences of which is greater than that value, to improve the prediction
    quality.
    * `feats_prune_coef=0` means "do not use feats";
    * `feats_prune_coef=None` means "use all feats";
    * default `feats_prune_coef=6`.
    """
    def __init__(self, feats_prune_coef=6):
        super().__init__('DEPREL', feats_prune_coef=feats_prune_coef)
        self._model_class = DeprelTaggerModel

    #@staticmethod
    def _preprocess_corpus(self, corpus):

        def window_right(sent, id_, ids, chains, window_len):
            link_ids = chains.get(id_, [])
            res = []
            for link_id in link_ids:
                idx = ids[link_id]
                token = sent[idx]
                if window_len == 1:
                    res.append([token])
                else:
                    for window_right_ in window_right(sent, link_id, ids,
                                                      chains, window_len - 1):
                        res.append([token] + window_right_)
            if not res:
                res = [[PAD_TOKEN] * window_len]
            return res

        def next_sent(sent, upper_sent, id_, ids, chains):
            link_ids = chains.get(id_, [])
            for link_id in link_ids:
                idx = ids[link_id]
                token = sent[idx]
                label = token['DEPREL']
                s = upper_sent[-WINDOW_LEFT:]
                s.append(token)
                for window_right_ in window_right(sent, link_id, ids,
                                                  chains, WINDOW_RIGHT):
                    s_ = ([PAD_TOKEN] * (WINDOW_LEFT + 1 - len(s))) \
                       + s[:] + window_right_
                    yield s_, idx, label
                for data_ in next_sent(sent, s, link_id, ids, chains):
                    yield data_

        res_corpus, labels, restore_data = [], [], []
        for i, sent in enumerate(corpus):
            if isinstance(sent, tuple):
                sent = sent[0]
            root_id, root_token, ids, chains = None, None, {}, {}
            for idx, token in enumerate(sent):
                id_, head = token['ID'], token['HEAD']
                if head:
                    if not token['FORM']:
                        token['FORM'] = PAD
                    if head == '0':
                        root_id, root_token = id_, token
                    ids[id_] = idx
                    chains.setdefault(head, []).append(id_)
            if root_id:
                for s, idx, label in next_sent(sent, [root_token],
                                               root_id, ids, chains):
                    res_corpus.append(s)
                    labels.append(label)
                    restore_data.append((i, idx))

        return res_corpus, labels, restore_data

    @staticmethod
    def _postprocess_corpus(corpus, labels, restore_data):
        for label, (i, idx) in zip(labels, restore_data):
            corpus[i][idx]['DEPREL'] = label
        return corpus

    @classmethod
    def _prepare_corpus(cls, corpus, fields, tags_to_remove=None):
        print()
        [print(x) for x in corpus[:5]]
        res = super()._prepare_corpus(corpus, fields,
                                      tags_to_remove=tags_to_remove)
        print()
        [print(list(x)) for x in zip(res[0][:5], res[1][:5], res[2][:5])]
        res = list(res)
        res[-1] = [x[WINDOW_LEFT] for x in res[-1]]
        print()
        [print(list(x)) for x in zip(res[0][:5], res[1][:5], res[2][:5])]
        return tuple(res)

    def train(self, save_as,
              device=None, epochs=None, min_epochs=0, bad_epochs=5,
              batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
              max_grad_norm=None, tags_to_remove=None,
              word_emb_type='bert', word_emb_model_device=None,
              word_emb_path=None, word_emb_tune_params=None,
              word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              upos_emb_dim=300, emb_out_dim=512, lstm_hidden_dim=256,
              lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
              bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
        """Creates and trains a feature tagger model.

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
        during which the selected **control_metric** does not improve) in a row.
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

        **word_transform_kwargs** (`dict`): keyword arguments for the
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
        labels. Default `upos_emb_dim=300`.

        **emb_out_dim** (`int`): output embedding dimensionality. Default
        `emb_out_dim=512`.

        **lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
        `lstm_hidden_dim=256`.

        **lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
        `lstm_layers=2`.

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
        args, kwargs = get_func_params(DeprelTagger.train, locals())

        if self._train_corpus:
            self._train_corpus, _, _ = \
                self._preprocess_corpus(self._train_corpus)
        if self._test_corpus:
            self._test_corpus, _, _ = \
                self._preprocess_corpus(self._test_corpus)

        return super().train(*args, **kwargs)
