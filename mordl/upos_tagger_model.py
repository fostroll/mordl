# -*- coding: utf-8 -*-
# MorDL project: UPOS tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from collections.abc import Iterable
from junky import get_func_params
from mordl.base_tagger_model import BaseTaggerModel
from mordl.defaults import CONFIG_ATTR


class UposTaggerModel(BaseTaggerModel):

    def __init__(self, labels_num, labels_pad_idx=None, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=1,
                 lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
                 bn3=True, do3=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = get_func_params(UposTaggerModel.__init__, locals())
        super().__init__(*args, **kwargs)
        setattr(self, CONFIG_ATTR, (args, kwargs))
