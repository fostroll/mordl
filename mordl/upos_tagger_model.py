# -*- coding: utf-8 -*-
# MorDL project: UPOS tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides UPOS tagger model inherited from `mordl.BaseTaggerModel`.
"""
from collections.abc import Iterable
from junky import get_func_params
from mordl.base_tagger_model import BaseTaggerModel
from mordl.defaults import CONFIG_ATTR


class UposTaggerModel(BaseTaggerModel):
    """Tagger class for `str` fields (like the *UPOS* field).

    Args:

    **labels_num** (`int`): number of target labels.

    **labels_pad_idx** (`int`): index of padding element in the label
    vocabulary.

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
