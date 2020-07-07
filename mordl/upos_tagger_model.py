# -*- coding: utf-8 -*-
# MorDL project: LSTM tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from collections.abc import Iterable
from junky import CharEmbeddingRNN, CharEmbeddingCNN, Masking
from mordl.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class UposTaggerModel(BaseModel):

    def __init__(self, tags_count, tags_pad_idx=None, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=1,
                 lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
                 bn3=True, do3=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        super().__init__(
            tags_count, tags_pad_idx=tags_pad_idx, vec_emb_dim=vec_emb_dim,
            alphabet_size=alphabet_size, char_pad_idx=char_pad_idx,
            rnn_emb_dim=rnn_emb_dim, cnn_emb_dim=cnn_emb_dim,
            cnn_kernels=cnn_kernels, emb_out_dim=emb_out_dim,
            lstm_hidden_dim=lstm_hidden_dim, lstm_layers=lstm_layers,
            lstm_do=lstm_do, bn1=bn1, do1=do1, bn2=bn2, do2=do2,
            bn3=bn3, do3=do3
        )
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

        self._bn1 = \
            nn.BatchNorm1d(num_features=vec_emb_dim
                                      + rnn_emb_dim
                                      + cnn_emb_dim) if bn1 else None
        self._do1 = nn.Dropout(p=do1) if do1 else None

        self._emb_fc_l = nn.Linear(
            in_features=vec_emb_dim + rnn_emb_dim + cnn_emb_dim,
            out_features=emb_out_dim
        )
        self._bn2 = \
            nn.BatchNorm1d(num_features=emb_out_dim) if bn2 else None
        self._do2 = nn.Dropout(p=do2) if do2 else None

        self._lstm_l = nn.LSTM(input_size=emb_out_dim,
                               hidden_size=lstm_hidden_dim,
                               num_layers=lstm_layers, batch_first=True,
                               dropout=lstm_do, bidirectional=True)
        self._T = nn.Linear(emb_out_dim, emb_out_dim)
        nn.init.constant_(self._T.bias, -1)

        self._bn3 = \
            nn.BatchNorm1d(num_features=lstm_hidden_dim * 2) if bn3 else None
        self._do3 = nn.Dropout(p=do3) if do3 else None

        self._out_l = nn.Linear(in_features=lstm_hidden_dim * 2,
                                out_features=tags_count)
        self._out_masking = \
            Masking(input_size=tags_count,
                    indices_to_highlight=tags_pad_idx,
                    batch_first=True) if tags_pad_idx is not None else None

    def forward(self, x, x_lens, x_ch, x_ch_lens):
        """
        x:    [batch[seq[w_idx + pad]]]
        lens: [seq_word_cnt]
        x_ch: [batch[seq[word[ch_idx + pad] + word[pad]]]]
        x_ch_lens: [seq[word_char_count]]
        """
        device = next(self.parameters()).device

        x_, x_e_rnn, x_e_cnn = [], None, None
        if self.vec_emb_dim:
            assert x.shape[2] == self.vec_emb_dim, \
                   'ERROR: invalid vector size: {} whereas vec_emb_dim = {}' \
                       .format(x.shape[2], self.vec_emb_dim)
            x_.append(x)
        if self._rnn_emb_l:
            x_e_rnn = self._rnn_emb_l(x_ch, x_ch_lens)
            x_.append(x_e_rnn)
        if self._cnn_emb_l:
            x_e_cnn = self._cnn_emb_l(x_ch, x_ch_lens)
            x_.append(x_e_cnn)

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
        x_, _ = self._lstm_l(x_)
        x_, _ = pad_packed_sequence(x_, batch_first=True)

        gate = torch.sigmoid(self._T(x))
        x = x_ * gate + x * (1 - gate)

        if self._bn3:
            x = x.transpose(1, 2)  # (N, L, C) to (N, C, L)
            x = self._bn3(x)
            x = x.transpose(1, 2)  # (N, C, L) to (N, L, C)
        if self._do3:
            x = self._do3(x)

        x = self._out_l(x)
        x = self._out_masking(x, x_lens)

        return x
