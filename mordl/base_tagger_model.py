# -*- coding: utf-8 -*-
# MorDL project: Base tagger model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a base model for MorDL taggers.
"""
from collections.abc import Iterable
from junky import BaseConfig, CharEmbeddingRNN, CharEmbeddingCNN, to_device
from mordl.base_model import BaseModel
from mordl.defaults import CONFIG_ATTR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BaseModelHeadConfig(BaseConfig):
    """
    The config with params to create `BaseModelHead` class.

    Args:

    **num_labels** (`int`): number of target labels.

    **labels_pad_idx** (`int`; default=-100): index of padding element in the
    label vocabulary. You can specify here your real index of the padding
    intent, but we recommend stay it as is (with default fake index) because
    in practice learning on padding increasing the resulting model
    performance.

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
    labels_pad_idx = -100
    vec_emb_dim = None

    alphabet_size = 0
    char_pad_idx = 0
    rnn_emb_dim = None
    cnn_emb_dim = None
    cnn_kernels = [1, 2, 3, 4, 5, 6]
    tag_emb_params = None
    emb_bn = True
    emb_do = .2
    final_emb_dim = 512

    pre_bn = True
    pre_do = .5
    lstm_layers = 1
    lstm_do = 0
    tran_layers = 0
    tran_heads = 8
    post_bn = True
    post_do = .4

    def __init__(self, num_labels, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels

class BaseTaggerModelHead(BaseModel):
    """
    A base model for MorDL taggers.

    Args:

    **config**: an instance of the `BaseModelHeadConfig` class or the dict
    that contains initialization data for it.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        assert config.final_emb_dim % 2 == 0, \
            'ERROR: `config.final_emb_dim` must be even ' \
           f"(now it's `{config.final_emb_dim}`."
        assert not (config.lstm_layers and config.tran_layers, \
            "ERROR: `config.lstm_layers` and `config.tran_layers` can't be " \
            'both set to non-zero.'

        cnn_kernels = config.cnn_kernels
        if isinstance(cnn_kernels, Iterable):
            config.cnn_kernels = list(cnn_kernels)

        if config.rnn_emb_dim:
            self._rnn_emb_l = \
                CharEmbeddingRNN(alphabet_size=config.alphabet_size,
                                 emb_dim=config.rnn_emb_dim,
                                 pad_idx=config.char_pad_idx)
        else:
            self._rnn_emb_l = None
            config.rnn_emb_dim = 0

        if config.cnn_emb_dim:
            self._cnn_emb_l = \
                CharEmbeddingCNN(alphabet_size=config.alphabet_size,
                                 emb_dim=config.cnn_emb_dim,
                                 pad_idx=config.char_pad_idx,
                                 kernels=config.cnn_kernels)
        else:
            self._cnn_emb_l = None
            config.cnn_emb_dim = 0

        self._tag_emb_l = self._tag_emb_ls = None
        tag_emb_dim = 0
        if config.tag_emb_params:
            if isinstance(tag_emb_params, dict):
                tag_emb_dim = config.tag_emb_params['dim'] or 0
                if tag_emb_dim:
                    self._tag_emb_l = \
                        nn.Embedding(
                            config.tag_emb_params['num'], tag_emb_dim,
                            padding_idx=config.tag_emb_params['pad_idx']
                        )
            else:
                self._tag_emb_ls = nn.ModuleList()
                for emb_params in config.tag_emb_params:
                    tag_emb_dim_ = emb_params['dim']
                    if tag_emb_dim_:
                        tag_emb_dim += tag_emb_dim_
                        self._tag_emb_ls.append(
                            nn.Embedding(emb_params['num'], tag_emb_dim_,
                                         padding_idx=emb_params['pad_idx'])
                        )
                    else:
                        self._tag_emb_ls.append(None)

        joint_emb_dim = config.vec_emb_dim + config.rnn_emb_dim \
                                           + config.cnn_emb_dim + tag_emb_dim

        # TODO: Wrap with nn.Sequential #####################################
        self._emb_bn = nn.BatchNorm1d(num_features=joint_emb_dim)
                           if config.emb_bn else \
                       None
        self._emb_do = nn.Dropout(p=config.emb_do) if config.emb_do else None

        self._emb_fc_l = nn.Linear(in_features=joint_emb_dim,
                                   out_features=final_emb_dim)
        self._pre_bn = nn.BatchNorm1d(num_features=config.final_emb_dim) \
                           if config.pre_bn else \
                       None
        self._pre_do = nn.Dropout(p=config.pre_do) if config.pre_do else None
        # TODO ##############################################################

        if config.lstm_layers > 0:
            self._lstm_l = nn.LSTM(
                input_size=config.final_emb_dim,
                hidden_size=config.final_emb_dim // 2,
                num_layers=config.lstm_layers, batch_first=True,
                dropout=config.lstm_do, bidirectional=True
            )
            self._T = nn.Linear(final_emb_dim, config.final_emb_dim)
            nn.init.constant_(self._T.bias, -1)
        else:
            self._lstm_l = None

        if config.tran_layers > 0:
            tran_enc_l = nn.TransformerEncoderLayer(
                config.final_emb_dim, config.tran_heads,
                dim_feedforward=2048, dropout=0.1, activation='relu',
                layer_norm_eps=1e-05
            )
            tran_norm_l = nn.LayerNorm(normalized_shape=config.final_emb_dim,
                                       eps=1e-6, elementwise_affine=True)
            self._tran_l = nn.TransformerEncoder(
                tran_enc_l, config.tran_layers, norm=tran_norm_l
            )
        else:
            self._tran_l = None

        # TODO: Wrap with nn.Sequential #####################################
        self._post_bn = \
            nn.BatchNorm1d(num_features=config.final_emb_dim) \
                if config.post_bn else \
            None
        self._post_do = nn.Dropout(p=config.post_do) if config.post_do else \
                        None

        self._out_l = nn.Linear(in_features=config.final_emb_dim,
                                out_features=config.num_labels)
        # TODO ##############################################################

        setattr(self, CONFIG_ATTR, (args, kwargs))

    def save(self, path, prefix=''):
        """Saves the current model.

        Args:

        **path** (`str`): the directory where to save.

        **prefix** (`str`; default is `None`): The default names for model
        files are `'config.json'` and `'state_dict.pt'`. But in some
        circumstances, these names may not be available. For such a case it is
        possible to specify a **prefix** for these names.
        """
        with open(os.path.join(path, prefix + 'config.json'), 'wt') as f:
            print(json.dumps(self.config, sort_keys=True, indent=4), file=f)
        #torch.save({x: y.cpu() for x, y in model.state_dict()},
        torch.save(model.state_dict(),
                   os.path.join(path, prefix + 'state_dict.pt'),
                   pickle_protocol=2)

    @classmethod
    def load(cls, path, prefix=''):
        """Loads a model.

        Args:

        **path** (`str`): the directory from where to load.

        **prefix** (`str`; default is `None`): The default names for model
        files are `'config.json'` and `'state_dict.pt'`. But in some
        circumstances, these names may have prefixes. For such a case it is
        possible to specify this **prefix** here.
        """
        with open(os.path.join(path, prefix + 'config.json'), 'rt') as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(
            torch.load(os.path.join(path, prefix + 'state_dict.pt'),
                       map_location='cpu')
        )
        return model

    def forward(self, x, x_lens, x_ch, x_ch_lens, *x_t labels=None):
        """
        x:    [N (batch), L (sentences), C (words + padding)]
        lens: number of words in sentences
        x_ch: [N, L, C (words + padding), S (characters + padding)]
        x_ch_lens: [L, number of characters in words]
        *x_t: [N, L, C (upos indices)], [N, L, C (feats indices)], ...
        """
        device = next(self.parameters()).device

        x_ = []
        if self.vec_emb_dim:
            assert x.shape[2] == self.vec_emb_dim, \
                   'ERROR: Invalid vector size: {} whereas vec_emb_dim = {}' \
                       .format(x.shape[2], self.vec_emb_dim)
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
        x.relu_(x)
        if self._pre_do:
            x = self._pre_do(x)

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
