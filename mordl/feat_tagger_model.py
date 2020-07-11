# -*- coding: utf-8 -*-
# MorDL project: NE tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from junky import get_func_params
from mordl.base_tagger_model import BaseTaggerModel


class FeatTaggerModel(BaseTaggerModel):

    def __init__(self, labels_num, labels_pad_idx=None, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 upos_emb_dim=60, upos_num=0, upos_pad_idx=0,
                 emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=1,
                 lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
                 bn3=True, do3=.4):
        args, kwargs = get_func_params(FeatTaggerModel.__init__, locals())
        kwargs_ = {x: y for x, y in kwargs.itemsif x not in [
            'upos_emb_dim', 'upos_num', 'upos_pad_idx'
        ]}
        if upos_emb_dim:
            kwargs_['tag_emb_params'] = {
                'dim': upos_emb_dim, 'num': upos_num, 'pad_idx': upos_pad_idx
            }
        super().__init__(*args, **kwargs_)
        setattr(self, CONFIG_ATTR, (args, kwargs))
