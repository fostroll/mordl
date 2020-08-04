# -*- coding: utf-8 -*-
# MorDL project: DEPREL tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides classes of DEPREL taggers.
"""
from copy import deepcopy
import itertools
import junky
from junky import get_func_params
from mordl import FeatTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.deprel_tagger_model import DeprelTaggerModel
import torch
from typing import Iterator


# LAS on *SynTagRus* (with supp_tagger of `DeprelSeqTagger` type) = 97.60
class DeprelTagger(FeatTagger):
    """
    A DEPREL tagger class.

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

    **supp_tagger**: an object of another DEPREL tagger which method
    `.predict()` has the same signature as `DeprelTagger.predict()` and no
    excess `'root'` tags in the return. Object of `DeprelSeqTagger` may be
    used here.
    """
    def __init__(self, feats_prune_coef=6, supp_tagger=None):
        super().__init__('DEPREL', feats_prune_coef=feats_prune_coef)
        self._supp_tagger = supp_tagger

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the DEPREL field of the corpus.

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

        Returns corpus with tags predicted in the DEPREL field.
        """
        assert self._ds is not None, \
               "ERROR: The tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(FeatTagger.predict, locals())

        kwargs['save_to'] = None

        corpus2 = None
        if self._supp_tagger:
            kwargs2 = kwargs.copy()
            kwargs2['with_orig'] = True
            corpus2 = self._supp_tagger.predict(*args, **kwargs2)

        corpus = super().predict(*args, **kwargs)

        def add_root(corpus):
            for sent in corpus:
                sent0 = sent[0] if isinstance(sent, tuple) \
                               and not isinstance(sent[0], tuple) \
                               and not isinstance(sent[1], tuple) else \
                        sent
                if isinstance(sent0, tuple):
                    sent0 = sent0[0]
                if corpus2:
                    sent2 = next(corpus2)[1]
                    if isinstance(sent2, tuple):
                        sent2 = sent2[0]
                for i, tok in enumerate(sent0):
                    if tok['HEAD'] == '0':
                        tok['DEPREL'] = 'root'
                    elif corpus2 and tok['DEPREL'] == 'root':
                        tok['DEPREL'] = sent2[i]['DEPREL']
                yield sent

        corpus = add_root(corpus)

        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)

        return corpus

NODES_UP, NODES_DOWN = 2, 2
PAD = '[PAD]'
PAD_TOKEN = {'ID': '0', 'FORM': PAD, 'LEMMA': PAD,
                         'UPOS': PAD, 'FEATS': {}, 'DEPREL': None}

class DeprelSeqTagger(FeatTagger):
    """
    A DEPREL tagger class with sequence model.

    **WARNING**: Works long, takes huge amoont of RAM, but accuracy is 
    less than of `DeprelTagger`. But may be useful as suppplement tagger
    for it.

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

    @staticmethod
    def _preprocess_corpus(corpus):

        def nodes_down(sent, id_, ids, nodes, span_len):
            link_ids = nodes.get(id_, [])
            res = []
            for link_id in link_ids:
                idx = ids[link_id]
                token = sent[idx]
                if span_len == 1:
                    res.append([token])
                else:
                    for nodes_down_ in nodes_down(sent, link_id, ids,
                                                  nodes, span_len - 1):
                        res.append([token] + nodes_down_)
            if not res:
                res = [[PAD_TOKEN] * span_len]
            return res

        def next_sent(sent, upper_sent, id_, ids, nodes):
            link_ids = nodes.get(id_, [])
            for link_id in link_ids:
                idx = ids[link_id]
                token = sent[idx]
                label = token['DEPREL']
                s = upper_sent[-NODES_UP:]
                s.append(token)
                for nodes_down_ in nodes_down(sent, link_id, ids,
                                              nodes, NODES_DOWN):
                    s_ = ([PAD_TOKEN] * (NODES_UP + 1 - len(s))) \
                       + s[:] + nodes_down_
                    yield s_, idx, label
                for data_ in next_sent(sent, s, link_id, ids, nodes):
                    yield data_

        res_corpus, labels, restore_data = [], [], []
        for i, sent in enumerate(corpus):
            if isinstance(sent, tuple):
                sent = sent[0]
            root_id, root_token, ids, nodes = None, None, {}, {}
            for idx, token in enumerate(sent):
                id_, head = token['ID'], token['HEAD']
                if head:
                    if not token['FORM']:
                        token['FORM'] = PAD
                    if head == '0':
                        root_id, root_token = id_, token
                    ids[id_] = idx
                    nodes.setdefault(head, []).append(id_)
            if root_id:
                for s, idx, label in next_sent(sent, [root_token],
                                               root_id, ids, nodes):
                    res_corpus.append(s)
                    labels.append(label)
                    restore_data.append((i, idx))

        return res_corpus, labels, restore_data

    @staticmethod
    def _postprocess_corpus(corpus, labels, restore_data):
        for label, (i, idx) in zip(labels, restore_data):
            token = corpus[i][idx]
            if not isinstance(token['DEPREL'], list):
                token['DEPREL'] = [label]
            else:
                token['DEPREL'].append(label)
        for sent in corpus:
            if isinstance(sent, tuple):
                sent = sent[0]
            for tok in sent:
                if tok['HEAD'] == '0':
                    tok['DEPREL'] = 'root'
                else:
                    deprel = tok['DEPREL']
                    if isinstance(deprel, list):
                        tok['DEPREL'] = max(set(deprel), key=deprel.count)
        return corpus

    def _prepare_corpus(self, corpus, fields, tags_to_remove=None):
        res = super()._prepare_corpus(corpus, fields,
                                      tags_to_remove=tags_to_remove)
        res = list(res)
        res[-1] = [x[NODES_UP] for x in res[-1]]
        return tuple(res)

    def load(self, name, device=None, dataset_device=None, log_file=LOG_FILE):
        """Loads feature tagger and dataset.

        Args:

        **name** (`str`): name of the previously saved internal state.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        override its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        args, kwargs = get_func_params(DeprelTagger.load, locals())
        super(self.__class__.__base__, self).load(DeprelTaggerModel,
                                                  *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the DEPREL field of the corpus.

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

        Returns corpus with tags predicted in the DEPREL field.
        """
        assert self._ds is not None, \
               "ERROR: The tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert self._model, \
               "ERROR: The tagger doesn't have a model. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(DeprelTagger.predict, locals())

        if self._feats_prune_coef != 0:
            key_vals = self._ds.get_dataset('t_0').transform_dict
            corpus = self._get_corpus(corpus, asis=True, log_file=log_file)
            corpus = self._transform_upos(corpus, key_vals=key_vals)

        field = 'DEPREL'
        add_fields = self._normalize_field_names('UPOS')

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

                orig_corpus = corpus_
                corpus_, _, restore_data = self._preprocess_corpus(corpus_)
                
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
                    res_corpus_ = deepcopy(orig_corpus)
                    res_corpus_ = \
                        self._postprocess_corpus(res_corpus_,
                                                 values, restore_data)
                    for orig_sentence, sentence in zip(orig_corpus,
                                                       res_corpus_):
                        yield sentence, orig_sentence
                else:
                    res_corpus_ = \
                        self._postprocess_corpus(orig_corpus,
                                                 values, restore_data)
                    for sentence in res_corpus_:
                        yield sentence

        corpus = process(corpus)

        if self._feats_prune_coef != 0:
            corpus = self._restore_upos(corpus)

        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

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
        """Creates and trains the DEPREL tagger model.

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
