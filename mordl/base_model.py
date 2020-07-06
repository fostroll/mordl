# -*- coding: utf-8 -*-
# MorDL project: Base model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import json
from mordl.utils import CONFIG_ATTR, LOG_FILE
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        setattr(self, CONFIG_ATTR, (args, kwargs))

    def save_config(self, f, log_file=LOG_FILE):
        config = []
        device = next(self._model.parameters()).device
        if device:
            config.append(str(device))
        cfg = getattr(self, CONFIG_ATTR, [])
        while cfg_ in cfg:
            if cfg_:
                config.append(cfg)
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
        while cfg in config:
            if isinstance(cfg, str) and not device:
                device = cfg
            elif isinstance(cfg, list) and not args:
                args = cfg
            elif isinstance(cfg, dict) and not kwargs:
                kwargs = cfg
        model = cls(*args, **kwargs)
        if device:
            model.to(device)
        if log_file:
            print('Model created', file=log_file)
        if state_dict_f:
            model.load_state_dict(state_dict_f, log_file=log_file)
        return model

    def save_state_dict(self, f, log_file=LOG_FILE):
        if log_file:
            print('Saving state_dict...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self.state_dict(), f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    def load_state_dict(self, f, log_file=LOG_FILE):
        if log_file:
            print('Loading state_dict...', end=' ', file=log_file)
            log_file.flush()
        super().load_state_dict(torch.load(f))
        self.eval()
        if log_file:
            print('done.', file=log_file)

    def save(self, f, log_file=LOG_FILE):
        if log_file:
            print('Saving model...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self, f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    @staticmethod
    def load(f, config_f=None, device=None, log_file=LOG_FILE):
        if log_file:
            print('Loading model...', end=' ', file=log_file)
            log_file.flush()
        model = torch.load(f, map_location=device)
        model.eval()
        if log_file:
            print('done.', file=log_file)
        return model

    @classmethod
    def create_model_for_train(cls, *args, lr=.0001, **kwargs):
        model = cls(*args, **kwargs)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        scheduler = None
        return model, criterion, optimizer, scheduler

    def adjust_model_for_tune(lr=.001, momentum=.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        scheduler = None
        return criterion, optimizer, scheduler
