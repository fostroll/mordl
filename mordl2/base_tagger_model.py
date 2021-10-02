# -*- coding: utf-8 -*-
# MorDL project: Base tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a base model for MorDL taggers.
"""
from collections import OrderedDict
from collections.abc import Iterable
from junky import CharEmbeddingRNN, CharEmbeddingCNN, Masking, \
                  get_func_params, to_device
from mordl2.base_model import BaseModel
from mordl2.defaults import CONFIG_ATTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BatchNorm(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BaseTaggerModel(BaseModel):
    """
    The base model for MorDL taggers.

    Args:

    **num_labels** (`int`): the number of target labels. Don't forget to add
    `1` for padding.

    **labels_pad_idx** (`int`; default=-100): the index of padding element in
    the label vocabulary. You can specify here your the real index of the
    padding intent, but we recommend to keep it as is (with default fake
    index) because in practice, learning on padding increasing the resulting
    model performance. If you stil want to experiment, along with specifying
    the real padding index, you may try not to add `1` to **num_labels** for
    the padding intent. The model then put random labels as tags for the
    padding part of the input, but they are ignored during the loss
    computation.

    **vec_emb_dim** (`int`; default is `None`): the incoming word-level
    embedding vector space dimensionality.

    **alphabet_size** (`int`): the length of character vocabulary for the
    internal character-level embedding layer. Relevant if either
    **rnn_emb_dim** or **cnn_emb_dim** is not `None`.

    **char_pad_idx** (`int`; default is `0`): the index of the padding element
    in the character vocabulary of the internal character-level embedding
    layer. Relevant if either **rnn_emb_dim** or **cnn_emb_dim** is not `None`.

    **rnn_emb_dim** (`int`; default is `None`): internal character RNN (LSTM)
    embedding dimensionality.

    **cnn_emb_dim** (`int`; default is `None`): internal character CNN
    embedding dimensionality. If `None`, the layer is skipped.

    **cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
    kernel sizes of the internal CNN embedding layer. Relevant if
    **cnn_emb_dim** is not `None`.

    **tag_emb_params** (`dict({'dim': int, 'num': int, 'pad_idx': int})` |
    `list([dict])`; default is `None`): params of internal embedding layers
    for additional `junky.dataset.TokenDataset` outputs. If `None`, the layers
    are not created.

    **emb_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied after the embedding concatenation.

    **emb_do** (`float`; default is '.2'): dropout rate after the embedding
    concatenation.

    **final_emb_dim** (`int`; default is `512`): the output dimesionality of
    the linear transformation applying to concatenated embeddings.

    **pre_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied before the main part of the algorithm.

    **pre_do** (`float`; default is '.5'): dropout rate before the main part
    of the algorithm.

    **lstm_layers** (`int`; default is `1`): the number of Bidirectional LSTM
    layers. If `0`, they are not created.

    **lstm_do** (`float`; default is `0`): dropout between LSTM layers. Only
    relevant, if `lstm_layers` > `1`.

    **tran_layers** (`int`; default is `0`): the number of Transformer Encoder
    layers. If `0`, they are not created.

    **tran_heads** (`int`; default is `8`): the number of attention heads of
    Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

    **post_bn** (`bool`; default is 'True'): whether batch normalization layer
    should be applied after the main part of the algorithm.

    **post_do** (`float`; default is '.4'): dropout rate after the main part
    of the algorithm.
    """
    def __init__(self, num_labels, labels_pad_idx=None, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 tag_emb_params=None, emb_bn=True, emb_do=.2,
                 final_emb_dim=512, pre_bn=True, pre_do=.5,
                 lstm_layers=1, lstm_do=0, tran_layers=0, tran_heads=8,
                 post_bn=True, post_do=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = get_func_params(BaseTaggerModel.__init__, locals())
        super().__init__(*args, **kwargs)

        assert final_emb_dim % 2 == 0, \
            'ERROR: `final_emb_dim` must be even ' \
           f"(now it's `{final_emb_dim}`)."
        assert not (lstm_layers and tran_layers), \
            "ERROR: `lstm_layers` and `tran_layers` can't be " \
            'both set to non-zero.'

        self.num_labels = num_labels
        self.labels_pad_idx = labels_pad_idx
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

        joint_emb_dim = vec_emb_dim + rnn_emb_dim + cnn_emb_dim + tag_emb_dim

        '''
        # TODO: Wrap with nn.Sequential #####################################
        self._emb_bn = \
            nn.BatchNorm1d(num_features=joint_emb_dim) if emb_bn else None
        self._emb_do = nn.Dropout(p=emb_do) if emb_do else None

        self._emb_fc_l = nn.Linear(in_features=joint_emb_dim,
                                   out_features=final_emb_dim)
        self._pre_bn = \
            nn.BatchNorm1d(num_features=final_emb_dim) if pre_bn else None
        self._pre_do = nn.Dropout(p=pre_do) if pre_do else None
        # TODO ##############################################################
        '''
        modules = OrderedDict
        if emb_bn:
            modules['emb_bn'] = BatchNorm(num_features=joint_emb_dim)
        if emb_do:
            modules['emb_do'] = nn.Dropout(p=emb_do)
        modules['emb_fc_l'] = nn.Linear(in_features=joint_emb_dim,
                                        out_features=final_emb_dim)
        if pre_bn:
            modules['pre_bn'] = BatchNorm(num_features=final_emb_dim)
        modules['pre_nl'] = nn.ReLU()
        if pre_do:
            modules['pre_do'] = nn.Dropout(p=pre_do)
        self._emb_seq_l = nn.Sequential(modules)
        # TODO ##############################################################

        if lstm_layers > 0:
            self._lstm_l = nn.LSTM(
                input_size=final_emb_dim,
                hidden_size=final_emb_dim // 2,
                num_layers=lstm_layers, batch_first=True,
                dropout=lstm_do, bidirectional=True
            )
            self._T = nn.Linear(final_emb_dim, final_emb_dim)
            nn.init.constant_(self._T.bias, -1)
        else:
            self._lstm_l = None

        if tran_layers > 0:
            tran_enc_l = nn.TransformerEncoderLayer(
                final_emb_dim, tran_heads,
                dim_feedforward=2048, dropout=0.1, activation='relu',
                layer_norm_eps=1e-05
            )
            tran_norm_l = nn.LayerNorm(normalized_shape=final_emb_dim,
                                       eps=1e-6, elementwise_affine=True)
            self._tran_l = nn.TransformerEncoder(
                tran_enc_l, tran_layers, norm=tran_norm_l
            )
        else:
            self._tran_l = None

        # TODO: Wrap with nn.Sequential #####################################
        self._post_bn = \
            nn.BatchNorm1d(num_features=final_emb_dim) if post_bn else None
        self._post_do = nn.Dropout(p=post_do) if post_do else None

        self._out_l = nn.Linear(in_features=final_emb_dim,
                                out_features=num_labels)
        # TODO ##############################################################

        setattr(self, CONFIG_ATTR, (args, kwargs))

    def forward(self, x, x_lens, x_ch, x_ch_lens, *x_t, labels=None):
        """
        x:         [N (batch), L (sentences), C (words + padding)]
        lens:      number of words in sentences
        x_ch:      [N, L, C (words + padding), S (characters + padding)]
        x_ch_lens: [L, number of characters in words]
        *x_t:      [N, L, C (upos indices)], [N, L, C (feats indices)], ...
        labels:    [N, L, C]
        """
        device = next(self.parameters()).device

        x_ = []
        vec_emb_dim = self.vec_emb_dim
        if vec_emb_dim:
            assert x.shape[2] == vec_emb_dim, \
                   'ERROR: Invalid vector size: ' \
                  f'`{x.shape[2]}` whereas `vec_emb_dim={vec_emb_dim}`'
            x_.append(to_device(x, device))
        if self._rnn_emb_l:
            x_.append(self._rnn_emb_l(to_device(x_ch, device), x_ch_lens))
        if self._cnn_emb_l:
            x_.append(self._cnn_emb_l(to_device(x_ch, device),
                                      to_device(x_ch_lens, device)))
        if self._tag_emb_l:
            x_.append(self._tag_emb_l(to_device(x_t[0], device)))
        elif self._tag_emb_ls:
            for l_, x_t_ in zip(self._tag_emb_ls, to_device(x_t, device)):
                if l_:
                    x_.append(l_(x_t_))

        x = x_[0] if len(x_) == 1 else torch.cat(x_, dim=-1)

        '''
        if self._emb_bn:
            x.transpose_(1, 2)  # (N, L, C) to (N, C, L)
            x = self._emb_bn(x)
            x.transpose_(1, 2)  # (N, C, L) to (N, L, C)
        if self._emb_do:
            x = self._emb_do(x)

        x = self._emb_fc_l(x)
        if self._pre_bn:
            x.transpose_(1, 2)  # (N, L, C) to (N, C, L)
            x = self._pre_bn(x)
            x.transpose_(1, 2)  # (N, C, L) to (N, L, C)
        x.relu_()
        if self._pre_do:
            x = self._pre_do(x)
        '''
        x = self._emb_seq_l(x)

        if self._lstm_l:
            x_ = pack_padded_sequence(x, x_lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
            x_, _ = self._lstm_l(x_)
            x_, _ = pad_packed_sequence(x_, batch_first=True)

            gate = torch.sigmoid(self._T(x))
            x = x_ * gate + x * (1 - gate)

        if self._tran_l:
            src_key_padding_mask = (
                torch.arange(x.shape[1], device=device).expand(x.shape[:-1])
             >= x_lens.view(1, -1).transpose(0, 1).expand(x.shape[:-1])
            )
            x.transpose_(0, 1)  # (N, L, C) to (L, N, C)
            x = self._tran_l(x, src_key_padding_mask=src_key_padding_mask)
            x.transpose_(0, 1)  # (L, N, C) to (N, L, C)

        if self._post_bn:
            x.transpose_(1, 2)  # (N, L, C) to (N, C, L)
            x = self._post_bn(x)
            x.transpose_(1, 2)  # (N, C, L) to (N, L, C)
        if self._post_do:
            x = self._post_do(x)

        logits = self._out_l(x)

        if labels is not None:
#             criterion = nn.CrossEntropyLoss().to(device)
#             loss = criterion(logits.flatten(end_dim=-2),
#                              labels.flatten(end_dim=-1))
            loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                   labels.view(-1),
                                   ignore_index=self.labels_pad_idx)
            logits = logits, loss

        return logits
