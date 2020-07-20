# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from Levenshtein import editops
from corpuscula import CorpusDict
from difflib import SequenceMatcher
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel

_OP_C_ASIS = 'asis'
_OP_C_TITLE = 'title'
_OP_C_LOWER = 'lower'


class LemmaTagger(BaseTagger):
    """"""

    def __init__(self, field='LEMMA', work_field=None):
        super().__init__()
        self._cdict = CorpusDict()
        self._orig_field = field
        self._field = work_field if work_field else field + 'd' 

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

    @staticmethod
    def _transform_upos(corpus):
        for sent in corpus:
            if isinstance(sent, tuple):
                sent = sent[0]
            isfirst = True
            for tok in sent:
                upos = tok['UPOS']
                if isfirst and tok['FORM'] and '-' in tok['ID'] \
                           and not tok['UPOS'].endswith('-first'):
                   tok['UPOS'] += '-first'
                   isfirst = False

    @staticmethod
    def _restore_upos(corpus, with_orig=False):
        def restore_upos(sentence):
            if isinstance(sent, tuple):
                sent = sent[0]
            for tok in sent:
                upos = tok['UPOS']
                if upos.endswith('-first'):
                    tok['UPOS'] = upos[:-6]
        [restore_upos(x) for x in corpus for x in (sent if with_orig else
                                                   [sent])]

    @classmethod
    def _find_affixes(cls, form, lemma):
        if form and lemma:
            a = cls.find_affixes(form, lemma)
            res = a[0], a[2], a[3], a[5]
        else:
            res = None,
        return res

    @staticmethod
    def get_editops(str_from, str_to, allow_replace=True, allow_copy=True):
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
        args, kwargs = get_func_params(LemmaTagger.load, locals())
        super().load(FeatTaggerModel, *args, **kwargs)
        self._cdict.restore_from(name + '.cdict.pickle', log_file=log_file)

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
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
                        token[self._orig_field] = \
                            apply_editops(form, token['UPOS'],
                                          token[self._orig_field],
                                          isfirst=isfirst)
                        isfirst = False
                yield sentence

        self._transfrorm_upos(corpus)
        corpus = process(
            super().predict(self._orig_field, 'UPOS', *args, **kwargs)
        )
        self._restore_upos(corpus)
        if save_to:
            self.save_conllu(corpus, save_to, log_file=None)
            corpus = self._get_corpus(save_to, asis=True, log_file=log_file)
        return corpus

    def evaluate(self, gold, test=None, feats=None, label=None,
                 batch_size=BATCH_SIZE, split=None, clone_ds=False,
                 log_file=LOG_FILE):
        assert not label or feats, \
            'ERROR: To evaluate the exact label you must specify its ' \
            'feat, too'
        assert not label or not feats \
                         or isinstance(feats, str) or len(feats) == 1, \
            'ERROR: To evaluate the exact label you must specify its own ' \
            'feat only'
        args, kwargs = get_func_params(LemmaTagger.evaluate, locals())
        field = self._orig_field
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
        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        self._cdict = CorpusDict(corpus=self._train_corpus,
                                 format='conllu_parsed', log_file=LOG_FILE)

        args, kwargs = get_func_params(LemmaTagger.train, locals())

        if log_file:
            print('Preliminary trainset preparation:', file=log_file)
            print('stage 1 of 3...', end=' ', file=log_file)
            log_file.flush()

        [x.update({self._field:
                       self._find_affixes(x['FORM'], x[self._orig_field])})
             for x in self._train_corpus for x in x]

        [x.update({self._field:
                       self._find_affixes(x['FORM'], x[self._orig_field])})
             for x in self._test_corpus for x in x]

        self._transform_upos(self._train_corpus)
        self._transform_upos(self._test_corpus)

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
                    lemma, affixes = tok[self._orig_field], tok[self._field]
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
                lemma, affixes = tok[self._orig_field], tok[self._field]
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
