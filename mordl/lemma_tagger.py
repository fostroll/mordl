# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a LemmaTagger class.
"""
from Levenshtein import editops
from corpuscula import CorpusDict
from difflib import SequenceMatcher, get_close_matches
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel

_OP_C_ASIS = 'asis'
_OP_C_TITLE = 'title'
_OP_C_LOWER = 'lower'


class LemmaTagger(BaseTagger):
    """
    A lemma tagger class.

    Args:

    Creates a `LemmaTagger` object.

    Args:

    **field** (`str`): the name of the field which needs to be predicted by
    the training tagger. May contain up to 3 elements, separated by a colon
    (`:`). Format is: `'<field name>:<feat name>:<replacement for None>'`. The
    replacement is used during the training time as a filler for a fields
    without     a value for that we could predict them, too. In the *CoNLL-U*
    format the     replacer is `'_'` sign, so we use it, too, as a default
    replacement. You'll hardly have a reason to change it. Examples:<br/> 
    `'UPOS'` - predict the *UPOS* field;<br/>
    `'FEATS:Animacy'` - predict only the *Animacy* feat of the *FEATS
    field;<br/>
    `'FEATS:Animacy:_O'` - likewise the above, but if feat value is `None`, it
    will be replaced by `'_O'` during training;<br/>
    `'XPOS::_O'` - predict the *XPOS* field and use `'_O'` as replacement for
    `None`.

    **feats_clip_coef** (`int`): feature clipping coefficient which allows to
    eliminate all features that have lower frequency than 
    `<most frequent feature frequency>` divided by `feats_clip_coef`.
    
    `feats_clip_coef=0` means "do not use feats"
    `feats_clip_coef=None` means "use all feats"
    """
    def __init__(self, field='LEMMA', feats_clip_coef=6):
        super().__init__()
        self._field = field
        self._feats_clip_coef = feats_clip_coef

    @staticmethod
    def find_affixes(form, lemma, lower=False):
        """Find the longest common part of the given **form** and **lemma**.

        :param lower: if `True` then return values will be always in lower
                      case
        :return: prefix, common part, suffix/flexion of **form**;
                 prefix, common part, suffix/flexion of **lemma**
        :rtype: `str`, `str`, `str`, `str`, `str`, `str`
        """
        if lower:
            lex = form = form.lower()
            lem = lemma = lemma.lower()
        else:
            lex = form.lower()
            lem = lemma.lower()
        a, b, size = SequenceMatcher(None, lex, lem, False) \
                         .find_longest_match(0, len(lex), 0, len(lem))
        return form[:a], form[a:a + size], form[a + size:], \
               lemma[:b], lemma[b:b + size], lemma[b + size:]

    def _transform_upos(self, corpus, key_vals=None):
        rel_feats = {}
        clip_coef = self._feats_clip_coef
        if clip_coef != 0:
            tags = self._cdict.get_tags_freq()
            if tags:
                thresh = tags[0][1] / clip_coef if clip_coef else 0
                tags = [x[0] for x in tags if x[1] > thresh]
            for tag in tags:
                feats_ = rel_feats[tag] = set()
                tag_feats = self._cdict.get_feats_freq(tag)
                if tag_feats:
                    thresh = tag_feats[0][1] / clip_coef if clip_coef else 0
                    [feats_.add(x[0]) for x in tag_feats if x[1] > thresh]

        for sent in corpus:
            if isinstance(sent, tuple):
                sent = sent[0]
            for tok in sent:
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

    @staticmethod
    def get_editops(str_from, str_to, allow_replace=True, allow_copy=True):
        """Get edit operations from `str_from` to `str_to` according to
        Levenstein distance. Supported edit operations: 'delete', 'insert',
        'replace', 'copy'.

        Args:

        **str_from** (`str`): source string.

        **str_to** (`str`): target string.

        **allow_replace** (`bool`): whether to allow **replace** edit
        operation.

        **allow_copy** (`bool`): whether to allow **copy** edit operation.
        """
        res = []
        for op, idx_dst, idx_src in editops(str_from, str_to):
            if op == 'delete':
                res.append(('d', idx_dst, None))
            else:
                ch_src = str_to[idx_src]
                if op == 'replace' and allow_replace:
                    res.append(('r', idx_dst, ch_src))
                elif op in ['insert', 'replace']:
                    op_prev, idx_prev, ch_prev = res[-1] if res else [0] * 3
                    if allow_copy and idx_prev \
                                  and str_from[idx_prev - 1] == ch_src \
                                  and (op_prev == 'c' or idx_prev != idx_dst):
                        res.append(('c', idx_dst, None))
                    else:
                        res.append(('i', idx_dst, str_to[idx_src]))
                    if op == 'replace':
                        res.append(('d', idx_dst, None))
                else:
                    raise ValueError("Unexpected operation code '{}'"
                                         .format(op))
        return tuple(res)

    @staticmethod
    def apply_editops(str_from, ops):
        """Apply edit operations to the source string.

        Args:

        **str_from** (`str`): source string to apply edit operations to.

        **ops** (`list([str])`): list of edit operations.
        """
        str_from = list(str_from)
        for op, idx, ch in reversed(ops):
            if op == 'd':
                del str_from[idx]
            elif op == 'i':
                str_from.insert(idx, ch)
            elif op == 'r':
                str_from[idx] = ch
            elif op == 'c':
                str_from.insert(idx, str_from[idx - 1])
        return ''.join(str_from)

    def load(self, name, device=None, dataset_device=None, log_file=LOG_FILE):
        """Loads tagger's internal state saved by its `.save()` method.

        Args:

        **name** (`str`): name of the internal state previously saved.

        **device**: a device for the loading model if you want to override its
        previously saved value.

        **dataset_device**: a device for the loading dataset if you want to
        override its previously saved value.

        **log_file**: a stream for info messages. Default is `sys.stdout`.
        """
        args, kwargs = get_func_params(LemmaTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        """Predicts tags in the `Lemma` field of the corpus.

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

        Returns corpus with tag predictions in the `Lemma` field.
        """
        assert self._ds is not None, \
               "ERROR: The tagger doesn't have a dataset. Call the train() " \
               'method first'
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(LemmaTagger.predict, locals())
        kwargs['save_to'] = None

        cdict = self._cdict
        yof = len([x for x in cdict._wforms if 'ё' in x])
        yol = len([x for x in cdict._lemmata if 'ё' in x])
        remove_yo = yol / yof < 10

        def apply_editops(str_from, upos, ops_t, isfirst):
            if str_from and ops_t not in [None, (None,)]:
                str_from_, coef = \
                    cdict.predict_lemma(str_from, upos, isfirst=isfirst)
                if coef >= .9:
                    str_from = str_from_
                else:
                    try:
                        ops_p, ops_s, ops_c = ops_t
                        str_from_ = ''.join(reversed(
                            self.apply_editops(reversed(
                                self.apply_editops(str_from, ops_p)
                            ), ops_s)
                        ))
                        if str_from_:
                            str_from = str_from_
                            if 'ё' in str_from:
                                if not cdict.lemma_isknown(str_from, upos):
                                    str_from_ = str_from.replace('ё', 'е')
                                    if remove_yo \
                                    or cdict.lemma_isknown(str_from_, upos):
                                        str_from = str_from_
                    except IndexError:
                        pass
                    if ops_c == _OP_C_LOWER:
                        str_from = str_from.lower()
                    elif ops_c == _OP_C_TITLE:
                        str_from = str_from.title()
            return str_from

        def process(corpus):
            for sentence in corpus:
                sentence_ = sentence[0] if with_orig else sentence
                if isinstance(sentence_, tuple):
                    sentence_ = sentence_[0]
                isfirst = True
                for token in sentence_:
                    id_, form = token['ID'], token['FORM']
                    if token and '-' not in id_:
                        token[self._field] = \
                            apply_editops(form, token['UPOS'],
                                          token[self._field], isfirst=isfirst)
                        isfirst = False
                yield sentence

        key_vals = self._ds.get_dataset('t_0').transform_dict
        corpus = self._get_corpus(corpus, asis=True, log_file=log_file)
        corpus = self._transform_upos(corpus, key_vals=key_vals)
        corpus = super().predict(self._field, 'UPOS', corpus, **kwargs)
        corpus = self._restore_upos(corpus)
        corpus = process(corpus)

        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, gold, test=None, batch_size=BATCH_SIZE, split=None,
                 clone_ds=False, log_file=LOG_FILE):
        """Evaluate predicitons on the development test set.

        Args:

        **gold** (`tuple(<sentences> <labels>)`): corpus with actual target
        tags.

        **test** (`tuple(<sentences> <labels>)`): corpus with predicted target
        tags. If `None`, predictions will be created on-the-fly based on the
        `gold` corpus.

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
        args, kwargs = get_func_params(LemmaTagger.evaluate, locals())
        return super().evaluate(self._field, *args, **kwargs)

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
        """Creates and trains a lemma prediction model.

        During training, the best model is saved after each successful epoch.

        Args:

        **save_as** (`str`): the name of the tagger using for save. As a
        result, 4 files will be created after training: two for tagger's model
        (config and state dict) and two for the dataset (config and the
        internal state). All file names are used **save_as** as prefix and
        their endings are: `.config.json` and `.pt` for the model;
        `_ds.config.json` and `_ds.pt` for the dataset.

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
        supported by the `junky.train()` method. Currently, options are:
        'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **max_grad_norm** (`float`): gradient clipping parameter, used with
        `torch.nn.utils.clip_grad_norm_()`.

        **tags_to_remove** (`list([str])|dict({str: list([str])})`): tags,
        tokens with those must be removed from the corpus. May be a `list` of
        tag names or a `dict` of `{<feat name>: [<feat value>, ...]}`. This
        argument may be used, for example, to remove some infrequent tags from
        the corpus. Note, that we remove the tokens from the train corpus as a
        whole, not just replace those tags to `None`.

        **word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v')
        embedding types.

        **word_emb_path** (`str`): path to word embeddings storage.

        **word_emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert'). `None` means
        **word_emb_model_device** = **device**

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

        args, kwargs = get_func_params(LemmaTagger.train, locals())

        self._cdict = CorpusDict(corpus=self._train_corpus,
                                 format='conllu_parsed', log_file=log_file)

        if log_file:
            print('\nPreliminary trainset preparation:', file=log_file)
            print('stage 1 of 3...', end=' ', file=log_file)
            log_file.flush()

        def find_affixes(form, lemma):
            if form and lemma:
                a = self.find_affixes(form, lemma)
                res = a[0], a[2], a[3], a[5]
            else:
                res = None,
            return lemma, res

        [x.update({self._field: find_affixes(x['FORM'], x[self._field])})
             for x in self._train_corpus for x in x]

        [x.update({self._field: find_affixes(x['FORM'], x[self._field])})
             for x in self._test_corpus for x in x]

        list(self._transform_upos(self._train_corpus))
        key_vals = set(x['UPOS'] for x in self._train_corpus for x in x
                           if x['FORM'] and x['UPOS'] and '-' not in x['ID'])
        list(self._transform_upos(self._test_corpus, key_vals))

        if log_file:
            print('done.', file=log_file)
            print('stage 2 of 3...', end=' ', file=log_file)
            log_file.flush()

        ops = []
        get_editops_kwargs = [{'allow_replace': x, 'allow_copy': y}
                                  for x in [True, False]
                                  for y in [True, False]]
        for kwargs_ in get_editops_kwargs:
            ops_ = []
            ops.append(ops_)
            for sent in self._train_corpus:
                for tok in sent:
                    lemma, affixes = tok[self._field]
                    if affixes != (None,):
                        f_p, f_s, l_p, l_s = affixes
                        ops_p = self.get_editops(f_p, l_p, **kwargs_)
                        ops_s = self.get_editops(''.join(reversed(f_s)),
                                                 ''.join(reversed(l_s)),
                                                 **kwargs_)
                        if lemma:
                            if lemma.istitle():
                                ops_c = _OP_C_TITLE
                            elif lemma.islower():
                                ops_c = _OP_C_LOWER
                            else:
                                ops_c = _OP_C_ASIS
                        else:
                            ops_c = _OP_C_ASIS
                        ops_.append((ops_p, ops_s, ops_c))
                    else:
                        ops_.append((None,))

        if log_file:
            print('done.', file=log_file)
            head_ = 'Lengths: ['
            print(head_, end='', file=log_file)
        num, idx, key_vals = len(self._train_corpus), -1, None
        for idx_, (ops_, kwargs_) in enumerate(zip(ops, get_editops_kwargs)):
            key_vals_ = set(ops_)
            num_ = len(key_vals_)
            if log_file:
                print('{}{} {}'
                          .format(',\n' + (' ' * len(head_)) if idx_ else '',
                                  num_, kwargs_),
                      end='', file=log_file)
            if num_ < num:
                num, idx, key_vals = num_, idx_, key_vals_
        if log_file:
            print(']', file=log_file)
            print('min = {}'.format(get_editops_kwargs[idx]), file=log_file)
            print('stage 3 of 3...', end=' ', file=log_file)
            log_file.flush()

        kwargs_ = get_editops_kwargs[idx]
        for sent in self._test_corpus:
            for tok in sent:
                lemma, affixes = tok[self._field]
                if affixes != (None,):
                    f_p, f_s, l_p, l_s = affixes
                    ops_p = self.get_editops(f_p, l_p, **kwargs_)
                    ops_s = self.get_editops(''.join(reversed(f_s)),
                                             ''.join(reversed(l_s)),
                                             **kwargs_)
                    if lemma:
                        if lemma.istitle():
                            ops_c = _OP_C_TITLE
                        elif lemma.islower():
                            ops_c = _OP_C_LOWER
                        else:
                            ops_c = _OP_C_ASIS
                    else:
                        ops_c = _OP_C_ASIS
                    tok[self._field] = ops_p, ops_s, ops_c
                else:
                    tok[self._field] = None,

        ops = iter(ops[idx])
        for sent in self._train_corpus:
            for tok in sent:
                ops_ = next(ops)
                tok[self._field] = ops_

        del ops
        if log_file:
            print('done.\n', file=log_file)

        [None if x[self._field] in key_vals else
         x.update({self._field: ((), (), x[self._field][2])})
             for x in self._test_corpus for x in x]

        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)
