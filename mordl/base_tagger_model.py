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
from junky import get_func_params, to_device
from junky.layers import CharEmbeddingRNN, CharEmbeddingCNN
from mordl.base_model import BaseModel
from mordl.defaults import CONFIG_ATTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BatchNorm(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class BaseTaggerModel(BaseModel):
    """
    The base model for MorDL taggers.

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

    **tag_emb_params** (`dict({'dim': int, 'num': int, 'pad_idx': int})` |
    `list([dict])`; default is `None`): params of internal embedding layers
    for additional `junky.dataset.TokenDataset` outputs. If `None`, the layers
    are not created.

    **emb_bn** (`bool`; default is `True`): whether batch normalization layer
    should be applied after the embedding concatenation.

    **emb_do** (`float`; default is `.2`): the dropout rate after the
    embedding concatenation.

    **final_emb_dim** (`int`; default is `512`): the output dimesionality of
    the linear transformation applying to concatenated embeddings.

    **pre_bn** (`bool`; default is `True`): whether batch normalization layer
    should be applied before the main part of the algorithm.

    **pre_do** (`float`; default is `.5`): the dropout rate before the main
    part of the algorithm.

    **lstm_layers** (`int`; default is `1`): the number of Bidirectional LSTM
    layers. If `None`, they are not created.

    **lstm_do** (`float`; default is `0`): the dropout between LSTM layers.
    Only relevant, if `lstm_layers` > `1`.

    **tran_layers** (`int`; default is `None`): the number of Transformer Encoder
    layers. If `None`, they are not created.

    **tran_heads** (`int`; default is `8`): the number of attention heads of
    Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

    **post_bn** (`bool`; default is `True`): whether batch normalization layer
    should be applied after the main part of the algorithm.

    **post_do** (`float`; default is `.4`): the dropout rate after the main
    part of the algorithm.
    """
    def __init__(self, num_labels, labels_pad_idx=-100, vec_emb_dim=None,
                 alphabet_size=0, char_pad_idx=0, rnn_emb_dim=None,
                 cnn_emb_dim=None, cnn_kernels=[1, 2, 3, 4, 5, 6],
                 tag_emb_params=None, emb_bn=True, emb_do=.2,
                 final_emb_dim=512, pre_bn=True, pre_do=.5,
                 lstm_layers=1, lstm_do=0, tran_layers=None, tran_heads=8,
                 post_bn=True, post_do=.4):
        if isinstance(cnn_kernels, Iterable):
            cnn_kernels = list(cnn_kernels)
        args, kwargs = get_func_params(BaseTaggerModel.__init__, locals())
        super().__init__(*args, **kwargs)

        assert final_emb_dim % 2 == 0, \
            'ERROR: `final_emb_dim` must be even ' \
           f"(now it's `{final_emb_dim}`)."

        self.num_labels = num_labels
        self.labels_pad_idx = labels_pad_idx
        if vec_emb_dim is None:
            vec_emb_dim = 0
        self.vec_emb_dim = vec_emb_dim

        if rnn_emb_dim:
            self.rnn_emb_l = \
                CharEmbeddingRNN(alphabet_size=alphabet_size,
                                 emb_dim=rnn_emb_dim,
                                 pad_idx=char_pad_idx)
        else:
            self.rnn_emb_l = None
            rnn_emb_dim = 0

        if cnn_emb_dim:
            self.cnn_emb_l = \
                CharEmbeddingCNN(alphabet_size=alphabet_size,
                                 emb_dim=cnn_emb_dim,
                                 pad_idx=char_pad_idx,
                                 kernels=cnn_kernels)
        else:
            self.cnn_emb_l = None
            cnn_emb_dim = 0

        self.tag_emb_l = self.tag_emb_ls = None
        tag_emb_dim = 0
        if tag_emb_params:
            if isinstance(tag_emb_params, dict):
                tag_emb_dim = tag_emb_params['dim'] or 0
                if tag_emb_dim:
                    self.tag_emb_l = \
                        nn.Embedding(tag_emb_params['num'], tag_emb_dim,
                                     padding_idx=tag_emb_params['pad_idx'])
            else:
                self.tag_emb_ls = nn.ModuleList()
                for emb_params in tag_emb_params:
                    tag_emb_dim_ = emb_params['dim']
                    if tag_emb_dim_:
                        tag_emb_dim += tag_emb_dim_
                        self.tag_emb_ls.append(
                            nn.Embedding(emb_params['num'], tag_emb_dim_,
                                         padding_idx=emb_params['pad_idx'])
                        )
                    else:
                        self.tag_emb_ls.append(None)

        joint_emb_dim = vec_emb_dim + rnn_emb_dim + cnn_emb_dim + tag_emb_dim
        assert joint_emb_dim, \
            'ERROR: At least one of `*_emb_dim` must be specified.'

        # PREPROCESS #########################################################
        modules = OrderedDict()
        if emb_bn:
            modules['emb_bn'] = BatchNorm(num_features=joint_emb_dim)
        if emb_do:
            modules['emb_do'] = nn.Dropout(p=emb_do)

        layers = []
        def add_layers(dim, new_dim):
            ls = []
            ls.append(('pre_fc{}_l',
                       nn.Linear(in_features=new_dim, out_features=dim)))
            if pre_bn:
                ls.append(('pre_bn{}', BatchNorm(num_features=dim)))
            ls.append(('pre_nl{}', nn.ReLU()))
            if pre_do:
                ls.append(('pre_do{}', nn.Dropout(p=pre_do)))
            layers.append(ls)

        dim = final_emb_dim
        while joint_emb_dim / dim > 2:
            new_dim = int(dim * 1.5)
            add_layers(dim, new_dim)
            dim = new_dim
        add_layers(dim, joint_emb_dim)
        for idx, layer in enumerate(reversed(layers)):
            for name, module in layer:
                modules[name.format(idx)] = module

        #modules['pre_fc_l'] = nn.Linear(in_features=dim,
        #                                out_features=final_emb_dim)
        #if pre_bn:
        #    modules['pre_bn'] = BatchNorm(num_features=final_emb_dim)
        #modules['pre_nl'] = nn.ReLU()
        #if pre_do:
        #    modules['pre_do'] = nn.Dropout(p=pre_do)
        self.pre_seq_l = nn.Sequential(modules)
        ######################################################################

        if lstm_layers:
            self.lstm_l = nn.LSTM(
                input_size=final_emb_dim,
                hidden_size=final_emb_dim // 2,
                num_layers=lstm_layers, batch_first=True,
                dropout=lstm_do, bidirectional=True
            )
            self.T = nn.Linear(final_emb_dim, final_emb_dim)
            nn.init.constant_(self.T.bias, -1)
        else:
            self.lstm_l = None

        if tran_layers:
            tran_enc_l = nn.TransformerEncoderLayer(
                final_emb_dim, tran_heads, dim_feedforward=2048,
                dropout=0.1, activation='relu'#, layer_norm_eps=1e-5
            )
            tran_norm_l = nn.LayerNorm(normalized_shape=final_emb_dim,
                                       eps=1e-6, elementwise_affine=True)
            self.tran_l = nn.TransformerEncoder(
                tran_enc_l, tran_layers, norm=tran_norm_l
            )
        else:
            self.tran_l = None

        # POSTPROCESS ########################################################
        modules = OrderedDict()
        if post_bn:
            modules['post_bn'] = BatchNorm(num_features=final_emb_dim)
        if post_do:
            modules['post_do'] = nn.Dropout(p=post_do)

        modules['out_fc_l'] = nn.Linear(in_features=final_emb_dim,
                                        out_features=num_labels)
        self.post_seq_l = nn.Sequential(modules)
        ######################################################################

        setattr(self, CONFIG_ATTR, (args, kwargs))

    def forward(self, x, x_lens, x_ch, x_ch_lens, *x_t, labels=None):
        """
        x:         [N (batch), L (sentences), C (words + padding)]
        lens:      number of words in sentences
        x_ch:      [N, L, C (words + padding), S (characters + padding)]
        x_ch_lens: [L, number of characters in words]
        *x_t:      [N, L, C (upos indices)], [N, L, C (xpos indices)], ...
        labels:    [N, L, C]

        Returns logits if label is `None`, (logits, loss) otherwise.
        """
        device = next(self.parameters()).device

        x_ = []
        vec_emb_dim = self.vec_emb_dim
        if vec_emb_dim:
            assert x.shape[2] == vec_emb_dim, \
                   'ERROR: Invalid vector size: ' \
                  f'`{x.shape[2]}` whereas `vec_emb_dim={vec_emb_dim}`'
            x_.append(to_device(x, device))
        if self.rnn_emb_l:
            x_.append(self.rnn_emb_l(to_device(x_ch, device), x_ch_lens))
        if self.cnn_emb_l:
            x_.append(self.cnn_emb_l(to_device(x_ch, device),
                                      to_device(x_ch_lens, device)))
        if self.tag_emb_l:
            x_.append(self.tag_emb_l(to_device(x_t[0], device)))
        elif self.tag_emb_ls:
            for l_, x_t_ in zip(self.tag_emb_ls, to_device(x_t, device)):
                if l_:
                    x_.append(l_(x_t_))

        x = x_[0] if len(x_) == 1 else torch.cat(x_, dim=-1)

        x = self.pre_seq_l(x)

        if self.lstm_l:
            x_ = pack_padded_sequence(x, x_lens.cpu(), batch_first=True,
                                      enforce_sorted=False)
            x_, _ = self.lstm_l(x_)
            x_, _ = pad_packed_sequence(x_, batch_first=True)

            gate = torch.sigmoid(self.T(x))
            x = x_ * gate + x * (1 - gate)

        if self.tran_l:
            src_key_padding_mask = (
                torch.arange(x.shape[1], device=device).expand(x.shape[:-1])
             >= x_lens.to(device).view(1, -1).transpose(0, 1)
                                             .expand(x.shape[:-1])
            )
            x.transpose_(0, 1)  # (N, L, C) to (L, N, C)
            x = self.tran_l(x, src_key_padding_mask=src_key_padding_mask)
            x.transpose_(0, 1)  # (L, N, C) to (N, L, C)

        logits = self.post_seq_l(x)

        if labels is not None:
#             criterion = nn.CrossEntropyLoss().to(device)
#             loss = criterion(logits.flatten(end_dim=-2),
#                              labels.flatten(end_dim=-1))
            loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                   labels.view(-1),
                                   ignore_index=self.labels_pad_idx)
            logits = logits, loss

        return logits
