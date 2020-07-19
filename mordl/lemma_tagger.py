# -*- coding: utf-8 -*-
# MorDL project: FEATS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from Levenshtein import editops
from corpuscula.utils import find_affixes
from junky import get_func_params
from mordl.base_tagger import BaseTagger
from mordl.defaults import BATCH_SIZE, LOG_FILE, TRAIN_BATCH_SIZE
from mordl.feat_tagger_model import FeatTaggerModel


class LemmaTagger(BaseTagger):
    """"""

    def __init__(self, field='LEMMA', work_field=None):
        super().__init__()
        self._orig_field = field
        self._field = work_field if work_field else field + 'd'
        self._field_a = field + 'a'
        self._field_u = field + 'u'
        self._field_c = field + 'c'
        self._field_p = field + 'p'
        self._field_s = field + 's'

    @staticmethod
    def _find_affixes(form, lemma):
        if form and lemma:
            a = find_affixes(form, lemma, lower=True)
            res = a[0], a[2], a[3], a[5]
        else:
            res = None,
        return res

    @staticmethod
    def _get_editops(str_from, str_to, allow_replace=True, allow_copy=True):
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
                    raise ValueError("Unexpected operation code '{}'".format(op))
        return tuple(res)

    @staticmethod
    def _apply_editops(str_from, ops):
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

    def predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
        assert not with_orig or save_to is None, \
               'ERROR: `with_orig` can be True only if save_to is None'
        args, kwargs = get_func_params(LemmaTagger.predict, locals())
        kwargs['save_to'] = None

        def apply_editops(str_from, ops_t):
                return ''.join(reversed(
                    self._apply_editops(reversed(
                        self._apply_editops(str_from, ops_t[0])
                    ), ops_t[1])
                )) if token[self._field] != (None,) else None

        def process(corpus):
            for sentence in corpus:
                sentence_ = sentence[0] if with_orig else sentence
                if isinstance(sentence_, tuple):
                    sentence_ = sentence_[0]
                for token in sentence_:
                    token[self._orig_field] = \
                        apply_editops(token['FORM'], token[self._orig_field]) \
                            if token[self._orig_field] != (None,) else None
                yield sentence

        corpus = process(
            super().predict(self._orig_field, 'UPOS', *args, **kwargs)
        )
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
                        ops_p = self._get_editops(f_p, l_p, **kwargs_)
                        ops_s = self._get_editops(''.join(reversed(f_s)),
                                                  ''.join(reversed(l_s)),
                                                  **kwargs_)
                        ops_c = bool(lemma and lemma.istitle())
                        ops_.append((ops_p, ops_s, ops_c))
                    else:
                        ops_.append((None,))

        if log_file:
            print('done. Lengths: [', end='', file=log_file)
            log_file.flush()
        num, idx, key_vals = len(self._train_corpus), -1, None
        for idx_, ops_ in enumerate(ops):
            key_vals_ = set(ops_)
            num_ = len(key_vals_)
            if log_file:
                print('{}{}'.format(', ' if idx_ else '', num_),
                      end='', file=log_file)
            if num_ < num:
                num, idx, key_vals = num_, idx_, key_vals_
        if log_file:
            print('], min = {}'.format(idx), file=log_file)
            print('stage 3 of 3...', file=log_file)

        '''
        if log_file:
            print('Lengths: [', end='', file=log_file)
        num, idx, key_vals = len(self._train_corpus), -1, None
        for idx_, ops_ in enumerate(ops):
            key_vals_ = set(ops_)
            num_ = len(key_vals_)
            if log_file:
                print('{}{}'.format(',\n          ' if idx_ else '', num_),
                      end='', file=log_file)
            if num_ < num:
                num, idx, key_vals = num_, idx_, key_vals_
        if log_file:
            print(']', file=log_file)
            print('min = {}'.format(get_editops_kwargs[idx]), file=log_file)
            print('...', end=' ', file=log_file)
            log_file.flush()
        '''

        kwargs_ = get_editops_kwargs[idx]
        for sent in self._test_corpus:
            for tok in sent:
                lemma, affixes = tok[self._orig_field], tok[self._field]
                if affixes != (None,):
                    f_p, f_s, l_p, l_s = affixes
                    ops_p = self._get_editops(f_p, l_p, **kwargs_)
                    ops_s = self._get_editops(''.join(reversed(f_s)),
                                              ''.join(reversed(l_s)),
                                              **kwargs_)
                    ops_c = bool(lemma and lemma.istitle())
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
         x.update({self._field: ((), ())})
             for x in self._test_corpus for x in x]

        return super().train(self._field, 'UPOS', FeatTaggerModel, 'upos',
                             *args, **kwargs)

    def train2(self, save_as,
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

        args, kwargs = get_func_params(LemmaTagger.train, locals())

        res = []

        if log_file:
            print('Preliminary trainset preparation...',
                  end=' ', file=log_file)
            log_file.flush()

        orig_field, field_a, field_u, field_c, field_p, field_s = \
            self._orig_field, self._field_a, self._field_u, \
            self._field_c, self._field_p, self._field_s

        for corp in (self._train_corpus, self._test_corpus):
            for sent in corp:
                isfirst = True
                for tok in sent:
                    form, lemma, upos = \
                        tok['FORM'], tok[orig_field], tok['UPOS']
                    tok[field_a] = self._find_affixes(form, lemma)
                    tok[field_c] = str(bool(lemma and lemma.istitle()))
                    if isfirst and form and '-' not in tok['ID']:
                        isfirst = False
                        tok[field_u] = upos + '-first'
                    else:
                        tok[field_u] = upos

        if log_file:
            print('done.', file=log_file)
            print('\n############ CAPITALIZATION ############\n',
                  file=log_file)

        res.append(super().train(field_c, field_u, FeatTaggerModel, 'upos',
                                 save_as + 'c', **kwargs))

        if log_file:
            print('\n############### PREFIXES ###############\n',
                  file=log_file)
            print('Preprocessing...', file=log_file)

        ops = []
        get_editops_kwargs = [{'allow_replace': x, 'allow_copy': y}
                                  for x in [True, False]
                                  for y in [True, False]]
        for kwargs_ in get_editops_kwargs:
            ops_ = []
            ops.append(ops_)
            for sent in self._train_corpus:
                for tok in sent:
                    form, affixes = tok['FORM'], tok[field_a]
                    if affixes != (None,):
                        f_, _, l_, _ = affixes
                        ops_.append(self._get_editops(f_, l_, **kwargs_))
                    else:
                        ops_.append(())

        if log_file:
            print('Lengths: [', end='', file=log_file)
        num, idx, key_vals = len(self._train_corpus), -1, None
        for idx_, ops_ in enumerate(ops):
            key_vals_ = set(ops_)
            num_ = len(key_vals_)
            if log_file:
                print('{}{}'.format(',\n          ' if idx_ else '', num_),
                      end='', file=log_file)
            if num_ < num:
                num, idx, key_vals = num_, idx_, key_vals_
        if log_file:
            print(']', file=log_file)
            print('min = {}'.format(get_editops_kwargs[idx]), file=log_file)
            print('...', end=' ', file=log_file)
            log_file.flush()

        kwargs_ = get_editops_kwargs[idx]
        for sent in self._test_corpus:
            for tok in sent:
                form, affixes = tok['FORM'], tok[field_a]
                if affixes != (None,):
                    f_, _, l_, _ = affixes
                    ops_ = self._get_editops(f_, l_, **kwargs_)
                    tok[field_p] = ops_ if ops_ in key_vals else ()
                else:
                    tok[field_p] = ()

        [x.update({field_p: next(ops)})
             for x in self._train_corpus for x in x]


        del ops
        if log_file:
            print('done.\n', file=log_file)

        res.append(super().train(field_p, field_u, FeatTaggerModel, 'upos',
                                 save_as + 'p', **kwargs))

        if log_file:
            print('\n############### SUFFIXES ###############\n',
                  file=log_file)
            print('Preprocessing...', file=log_file)

        ops = []
        get_editops_kwargs = [{'allow_replace': x, 'allow_copy': y}
                                  for x in [True, False]
                                  for y in [True, False]]
        for kwargs_ in get_editops_kwargs:
            ops_ = []
            ops.append(ops_)
            for sent in self._train_corpus:
                for tok in sent:
                    form, affixes = tok['FORM'], tok[field_a]
                    if affixes != (None,):
                        f_, _, l_, _ = affixes
                        ops_.append(self._get_editops(''.join(reversed(f_)),
                                                      ''.join(reversed(l_)),
                                                      **kwargs_))
                    else:
                        ops_.append(())

        if log_file:
            print('Lengths: [', end='', file=log_file)
        num, idx, key_vals = len(self._train_corpus), -1, None
        for idx_, ops_ in enumerate(ops):
            key_vals_ = set(ops_)
            num_ = len(key_vals_)
            if log_file:
                print('{}{}'.format(',\n          ' if idx_ else '', num_),
                      end='', file=log_file)
            if num_ < num:
                num, idx, key_vals = num_, idx_, key_vals_
        if log_file:
            print(']', file=log_file)
            print('min = {}'.format(get_editops_kwargs[idx]), file=log_file)
            print('...', end=' ', file=log_file)
            log_file.flush()

        kwargs_ = get_editops_kwargs[idx]
        for sent in self._test_corpus:
            for tok in sent:
                form, affixes = tok['FORM'], tok[field_a]
                if affixes != (None,):
                    f_, _, l_, _ = affixes
                    ops_ = self._get_editops(''.join(reversed(f_)),
                                              ''.join(reversed(l_)),
                                              **kwargs_)
                    tok[field_s] = ops_ if ops_ in key_vals else ()
                else:
                    tok[field_s] = ()

        [x.update({field_s: next(ops)})
             for x in self._train_corpus for x in x]

        del ops
        if log_file:
            print('done.\n', file=log_file)

        res.append(super().train(field_s, field_u, FeatTaggerModel, 'upos',
                                 save_as + 's', **kwargs))
