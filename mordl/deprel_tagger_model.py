# -*- coding: utf-8 -*-
# MorDL project: NE tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from collections.abc import Iterable
from junky import get_func_params
from mordl.base_tagger_sequence_model import BaseTaggerSequenceModel
from mordl.defaults import CONFIG_ATTR


class DeprelTaggerModel(BaseTaggerSequenceModel):
    """Named entity tagger class.

    Args:

    **labels_num** (`int`): number of target labels.

    **vec_emb_dim** (`int`): word-level embedding vector space dimensionality.
     If `None`, the layer is skipped.

    **alphabet_size** (`int`): length of character vocabulary. Relevant with
    not `None` **rnn_emb_dim** or **cnn_emb_dim**.

    **char_pad_idx** (`int`): index of padding element in the character
    vocabulary. Relevant with not `None` **rnn_emb_dim** or **cnn_emb_dim**.

    **rnn_emb_dim** (`int`): character RNN (LSTM) embedding dimensionality. If
    `None`, the layer is skipped.

    **cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
    `None`, the layer is skipped.

    **cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
    `cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
    **cnn_emb_dim**.

    **upos_emb_dim** (`int`): auxiliary UPOS label embedding dimensionality.
    Default `upos_emb_dim=200`.

    **upos_num** (`int`): length of UPOS vocabulary.

    **upos_pad_idx** (`int`): index of padding element in the UPOS vocabulary.

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
    """
    def __init__(self, labels_num, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 upos_emb_dim=200, upos_num=0, upos_pad_idx=0,
                 emb_out_dim=512, lstm_bidirectional=True,
                 lstm_hidden_dim=256, lstm_layers=1, lstm_do=0, bn1=True,
                 do1=.2, bn2=True, do2=.5, bn3=True, do3=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = get_func_params(DeprelTaggerModel.__init__, locals())
        kwargs_ = {x: y for x, y in kwargs.items() if x not in [
            'upos_emb_dim', 'upos_num', 'upos_pad_idx'
        ]}
        if upos_emb_dim:
            kwargs_['tag_emb_params'] = {
                'dim': upos_emb_dim, 'num': upos_num, 'pad_idx': upos_pad_idx
            }
        super().__init__(*args, **kwargs_)
        setattr(self, CONFIG_ATTR, (args, kwargs))
