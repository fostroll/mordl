# -*- coding: utf-8 -*-
# MorDL project: Base model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a base class for MorDL models.
"""
import json
from mordl.defaults import CONFIG_ATTR, LOG_FILE
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    The base class for MorDL models.

    Args:

    **\*args**: any mordl model args.

    **\*\*kwargs**: any mordl model keyword args.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        setattr(self, CONFIG_ATTR, (args, kwargs))

    def save_config(self, f, log_file=LOG_FILE):
        """Saves config in the specified file.

        Args:

        **f** (`str` | `file`): the file where the config should be saved.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        config = list(getattr(self, CONFIG_ATTR, []))
        device = next(self.parameters()).device
        if device:
            config.insert(0, str(device))
        need_close = False
        if isinstance(f, str):
            f = open(f, 'wt', encoding='utf-8')
            need_close = True
        try:
            print(json.dumps(config, sort_keys=True, indent=4), file=f)
        finally:
            if need_close:
                f.close()
        if log_file:
            print('Config saved', file=log_file)

    @classmethod
    def create_from_config(cls, f, state_dict_f=None, device=None,
                           log_file=LOG_FILE):
        """Creates model with parameters and a state dictionary previouly
        saved in the specified files.

        Args:

        **f** (`str` | `file`): the config file for the model creation.

        **state_dict_f** (`str` | `file`; default is `None`): the state
        dictionary file of the previously saved PyTorch model. If `None`, the
        model is initialized by default.

        **device** (`str`; default is `None`): the device where the model
        should be loaded after it's created, e.g. `'cuda:2'`. If `None`, the
        model is loaded to CPU.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        need_close = False
        if isinstance(f, str):
            f = open(f, 'rt', encoding='utf-8')
            need_close = True
        try:
            config = json.loads(f.read())
        finally:
            if need_close:
                f.close()
        args, kwargs = [], {}
        for cfg in config:
            if isinstance(cfg, str) and not device:
                device = cfg
            elif isinstance(cfg, list) and not args:
                args = cfg
            elif isinstance(cfg, dict) and not kwargs:
                kwargs = cfg
        if log_file:
            print('Creating model...', end=' ', file=log_file)
            log_file.flush()
        '''### backward compatibility ###########################################
        for old_kwarg, new_kwarg in zip(
            ['bn1', 'bn2', 'bn3', 'do1', 'do2', 'do3', 'emb_out_dim'],
            ['emb_bn', 'pre_bn', 'post_bn', 'emb_do', 'pre_do', 'post_do',
             'final_emb_dim']
        ):
            if old_kwarg in kwargs:
                assert new_kwarg not in kwargs, \
                   f'ERROR: The parameter {old_kwarg} in the config file ' \
                   f"can't be used together with {new_kwarg}."
            kwargs[new_kwarg] = kwargs[old_kwarg]
            del kwargs[old_kwarg]
        if 'lstm_hidden_dim' in kwargs:
            assert kwargs['lstm_hidden_dim'] \
                == kwargs['final_emb_dim'] // 2, \
               f'ERROR: Models with `lstm_hidden_dim` not equals to ' \
               f'`emb_out_dim // 2` are not supported anymore.'
            del kwargs['lstm_hidden_dim']
        ######################################################################'''
        model = cls(*args, **kwargs)
        if device:
            model.to(device)
        if log_file:
            print('done.', file=log_file)
        if state_dict_f:
            model.load_state_dict(state_dict_f, log_file=log_file)
        return model

    def save_state_dict(self, f, log_file=LOG_FILE):
        """Saves the PyTorch model's state dictionary to a file for further use
        in the inference.

        Args:

        **f** (`str` | `file`): the file where the state dictionary should be
        saved.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        if log_file:
            print('Saving state_dict...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self.state_dict(), f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    def load_state_dict(self, f, log_file=LOG_FILE):
        """Loads the previously saved PyTorch model's state dictionary for
        the inference.

        Args:

        **f** (`str` | `file`): the file from where the state dictionary
        should be loaded.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        if log_file:
            print('Loading state_dict...', end=' ', file=log_file)
            log_file.flush()
        device = next(self.parameters()).device
        super().load_state_dict(torch.load(f, map_location=device))
        self.eval()
        if log_file:
            print('done.', file=log_file)

    def save(self, f, log_file=LOG_FILE):
        """Saves the full model configuration to the single file.

        Args:

        **f** (`str` | `file`): the file where the model should be saved.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        if log_file:
            print('Saving model...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self, f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    @staticmethod
    def load(f, device=None, log_file=LOG_FILE):
        """Loads previously saved model configuration.

        Args:

        **f** (`str` | `file`): the file from where the model should be
        loaded.

        **device** (`str`; default is `None`): the device where the model
        should be loaded, e.g. `'cuda:2'`. If `None`, the model is loaded to
        CPU.

        **log_file** (`file`; default is `sys.stdout`): the stream for info
        messages.
        """
        if log_file:
            print('Loading model...', end=' ', file=log_file)
            log_file.flush()
        model = torch.load(f, map_location=device)
        model.eval()
        if log_file:
            print('done.', file=log_file)
        return model

    def adjust_model_for_train(self, lr=.0001, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=0, amsgrad=False,
                               labels_pad_idx=-100):
        """Creates model, criterion, optimizer and scheduler for training.
        Adam optimizer is used to train the model.

        Args:

        **lr**, **betas**, **eps**, **weight_decay**, **amsgrad**: the
        hyperparams for the torch.optim.Adam optimizer.

        **labels_pad_idx** (`int`; default=-100): the index of padding element
        in the label vocabulary.
        """
        criterion = nn.CrossEntropyLoss(ignore_index=labels_pad_idx)
        optimizer = torch.optim.Adam(
            self.parameters(), lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        scheduler = None
        return criterion, optimizer, scheduler

    def adjust_model_for_tune(self, lr=.001, momentum=.9, weight_decay=0,
                              dampening=0, nesterov=False,
                              labels_pad_idx=-100):
        """Ajusts model for post-train finetuning. Optimizer is changed to
        SGD to finetune the model.

        Args:

        **lr**, **momentum**, **weight_decay**, **dampening**, **nesterov**:
        the hyperparams for the torch.optim.SGD optimizer.

        **labels_pad_idx** (`int`; default=-100): the index of padding element
        in the label vocabulary.
        """
        criterion = nn.CrossEntropyLoss(ignore_index=labels_pad_idx)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=lr, momentum=momentum,
            weight_decay=weight_decay, dampening=dampening, nesterov=nesterov
        )
        scheduler = None
        return criterion, optimizer, scheduler
