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
    """
    The tagger class for `str` fields (like the *UPOS* field).

    Args:

    **num_labels** (`int`): the number of target labels. Don't forget to add
    `1` for padding.

    **labels_pad_idx** (`int`; default=-100): the index of padding element in
    the label vocabulary. You can specify here the real index of the padding
    intent, but we recommend to keep it as is (with default fake index)
    because in practice, learning on padding increasing the resulting model
    performance. If you stil want to experiment, along with specifying the
    real padding index, you may try not to add `1` to **num_labels** for the
    padding intent. The model then put random labels as tags for the padding
    part of the input, but they are ignored during the loss computation.

    **vec_emb_dim** (`int`; default is `None`): the incoming word-level
    embedding vector space dimensionality.

    **alphabet_size** (`int`): the length of character vocabulary for the
    internal character-level embedding layer. Relevant if either
    **rnn_emb_dim** or **cnn_emb_dim** is not `None`.

    **char_pad_idx** (`int`; default is `0`): the index of the padding element
    in the character vocabulary of the internal character-level embedding
    layer. Relevant if either **rnn_emb_dim** or **cnn_emb_dim** is not `None`.

    **rnn_emb_dim** (`int`; default is `None`): the internal character RNN
    (LSTM) embedding dimensionality. If `None`, the layer is skipped.

    **cnn_emb_dim** (`int`; default is `None`): the internal character CNN
    embedding dimensionality. If `None`, the layer is skipped.

    **cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
    kernel sizes of the internal CNN embedding layer. Relevant if
    **cnn_emb_dim** is not `None`.

    **emb_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied after the embedding concatenation.

    **emb_do** (`float`; default is '.2'): the dropout rate after the
    embedding concatenation.

    **final_emb_dim** (`int`; default is `512`): the output dimesionality of
    the linear transformation applying to concatenated embeddings.

    **pre_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied before the main part of the algorithm.

    **pre_do** (`float`; default is '.5'): the dropout rate before the main
    part of the algorithm.

    **lstm_layers** (`int`; default is `1`): the number of Bidirectional LSTM
    layers. If `None`, they are not created.

    **lstm_do** (`float`; default is `0`): the dropout between LSTM layers.
    Only relevant, if `lstm_layers` > `1`.

    **tran_layers** (`int`; default is `None`): the number of Transformer
    Encoder layers. If `None`, they are not created.

    **tran_heads** (`int`; default is `8`): the number of attention heads of
    Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

    **post_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied after the main part of the algorithm.

    **post_do** (`float`; default is '.4'): the dropout rate after the main
    part of the algorithm.
    """
    def __init__(self, num_labels, labels_pad_idx=-100, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 emb_bn=True, emb_do=.2,
                 final_emb_dim=512, pre_bn=True, pre_do=.5,
                 lstm_layers=1, lstm_do=0, tran_layers=None, tran_heads=8,
                 post_bn=True, post_do=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = get_func_params(UposTaggerModel.__init__, locals())
        super().__init__(*args, **kwargs)
        setattr(self, CONFIG_ATTR, (args, kwargs))
