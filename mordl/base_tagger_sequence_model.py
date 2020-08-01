# -*- coding: utf-8 -*-
# MorDL project: Base tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a base model for MorDL taggers.
"""
from collections.abc import Iterable
from junky import CharEmbeddingRNN, CharEmbeddingCNN, Masking, get_func_params
from mordl.base_model import BaseModel
from mordl.defaults import CONFIG_ATTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class BaseTaggerSequenceModel(BaseModel):
    """
    A base model for MorDL taggers.

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

    **tag_emb_params** (`dict({'dim': int, 'num': int, 'pad_idx': int})` |
    `list([dict])`): params of the embedding layers for additional
    `junky.dataset.TokenDataset` outputs. If `None`, the layers are not
    created.

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
                 tag_emb_params=None, emb_out_dim=512,
                 lstm_bidirectional=True, lstm_hidden_dim=256, lstm_layers=1,
                 lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
                 bn3=True, do3=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = \
            get_func_params(BaseTaggerSequenceModel.__init__, locals())
        super().__init__(*args, **kwargs)

        self.vec_emb_dim = vec_emb_dim

        if rnn_emb_dim:
            self._rnn_emb_l = \
                CharEmbeddingRNN(alphabet_size=alphabet_size,
                                 emb_dim=rnn_emb_dim,
                                 pad_idx=char_pad_idx)
        else:
            self._rnn_emb_l = None
            rnn_emb_dim = 0

        if cnn_emb_dim:
            self._cnn_emb_l = \
                CharEmbeddingCNN(alphabet_size=alphabet_size,
                                 emb_dim=cnn_emb_dim,
                                 pad_idx=char_pad_idx,
                                 kernels=cnn_kernels)
        else:
            self._cnn_emb_l = None
            cnn_emb_dim = 0

        self._tag_emb_l = self._tag_emb_ls = None
        tag_emb_dim = 0
        if tag_emb_params:
            if isinstance(tag_emb_params, dict):
                tag_emb_dim = tag_emb_params['dim'] or 0
                if tag_emb_dim:
                    self._tag_emb_l = \
                        nn.Embedding(tag_emb_params['num'], tag_emb_dim,
                                     padding_idx=tag_emb_params['pad_idx'])
            else:
                self._tag_emb_ls = nn.ModuleList()
                for emb_params in tag_emb_params:
                    tag_emb_dim_ = emb_params['dim']
                    if tag_emb_dim_:
                        tag_emb_dim += tag_emb_dim_
                        self._tag_emb_ls.append(
                            nn.Embedding(emb_params['num'], tag_emb_dim_,
                                         padding_idx=emb_params['pad_idx'])
                        )
                    else:
                        self._tag_emb_ls.append(None)

        self._bn1 = \
            nn.BatchNorm1d(num_features=vec_emb_dim
                                      + rnn_emb_dim
                                      + cnn_emb_dim
                                      + tag_emb_dim) if bn1 else None
        self._do1 = nn.Dropout(p=do1) if do1 else None

        self._emb_fc_l = nn.Linear(
            in_features=vec_emb_dim + rnn_emb_dim + cnn_emb_dim
                                    + tag_emb_dim,
            out_features=emb_out_dim
        )
        self._bn2 = \
            nn.BatchNorm1d(num_features=emb_out_dim) if bn2 else None
        self._do2 = nn.Dropout(p=do2) if do2 else None

        self._lstm_l = nn.LSTM(input_size=emb_out_dim,
                               hidden_size=lstm_hidden_dim,
                               num_layers=lstm_layers, batch_first=True,
                               dropout=lstm_do,
                               bidirectional=lstm_bidirectional)
        if lstm_bidirectional:
            lstm_hidden_dim *= 2

        self._bn3 = \
            nn.BatchNorm1d(num_features=lstm_hidden_dim) if bn3 else None
        self._do3 = nn.Dropout(p=do3) if do3 else None

        self._out_l = nn.Linear(in_features=lstm_hidden_dim,
                                out_features=labels_num)

        setattr(self, CONFIG_ATTR, (args, kwargs))

    def forward(self, x, x_lens, x_ch, x_ch_lens, *x_t):
        """
        x:    [batch[seq[w_idx + pad]]]
        lens: [seq_word_cnt]
        x_ch: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        x_ch_lens: [seq[word_char_count]]
        *x_t:  [batch[seq[upos_idx]]], ...
        """
        device = next(self.parameters()).device

        x_ = []
        if self.vec_emb_dim:
            assert x.shape[2] == self.vec_emb_dim, \
                   'ERROR: Invalid vector size: {} whereas vec_emb_dim = {}' \
                       .format(x.shape[2], self.vec_emb_dim)
            x_.append(x)
        if self._rnn_emb_l:
            x_.append(self._rnn_emb_l(x_ch, x_ch_lens))
        if self._cnn_emb_l:
            x_.append(self._cnn_emb_l(x_ch, x_ch_lens))
        if self._tag_emb_l:
            x_.append(self._tag_emb_l(x_t[0]))
        elif self._tag_emb_ls:
            for l_, x_t_ in zip(self._tag_emb_ls, x_t):
                if l_:
                    x_.append(l_(x_t_))

        x = x_[0] if len(x_) == 1 else torch.cat(x_, dim=-1)

        if self._bn1:
            x = x.transpose(1, 2)  # (N, L, C) to (N, C, L)
            x = self._bn1(x)
            x = x.transpose(1, 2)  # (N, C, L) to (N, L, C)
        if self._do1:
            x = self._do1(x)

        x = self._emb_fc_l(x)
        if self._bn2:
            x = x.transpose(1, 2)  # (N, L, C) to (N, C, L)
            x = self._bn2(x)
            x = x.transpose(1, 2)  # (N, C, L) to (N, L, C)
        x = F.relu(x)
        if self._do2:
            x = self._do2(x)

        x_ = pack_padded_sequence(x, x_lens, batch_first=True,
                                  enforce_sorted=False)
        _, (h_n, _) = self._lstm_l(x_)
        # h_n.shape => [batch size, num layers * num directions, hidden size]

        x = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) \
                if self._lstm_l.bidirectional else \
            h_n[-1, :, :]
        # x.shape => [batch size, hidden size]

        if self._bn3:
            x = self._bn3(x)

        if self._do3:
            x = self._do3(x)

        x = self._out_l(x)

        return x
