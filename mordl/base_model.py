# -*- coding: utf-8 -*-
# MorDL project: Base model
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import json
from mordl.defaults import CONFIG_ATTR, LOG_FILE
import torch
import torch.nn as nn


class BaseModel(nn.Module):

    """
    Base class for mordl models.
    
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
    
    **f**: name of the file where config will be saved.
    
    **log_file**: name of the log file to save logs. If not specified, 
    logs are printed to stdout.
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
        """Creates model from a previouly saved model's state dictionary.
        
        Args:
        
        **f**: the name of the file from where config will be loaded.
        
        **state_dict_f**: name of the state dictionary file 
        of the previously saved PyTorch model.
        
        **device**: device where the model will be loaded, 
        e.g. `torch.device('cuda:2')`. By default, the model is loaded to CPU.
        
        **log_file**: name of the log file to save logs. If not specified, 
        logs are printed to stdout.
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
        model = cls(*args, **kwargs)
        if device:
            model.to(device)
        if log_file:
            print('done.', file=log_file)
        if state_dict_f:
            model.load_state_dict(state_dict_f, log_file=log_file)
        return model

    def save_state_dict(self, f, log_file=LOG_FILE):
        """Saves Pytorch model's state dictionary to a file to further use 
        for model inference.
        
        Args:
        
        **f**: the name of the file where state dictionary will be saved.
        
        **log_file**: name of the log file to save logs. If not specified, 
        logs are printed to stdout.
        """
    
        if log_file:
            print('Saving state_dict...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self.state_dict(), f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    def load_state_dict(self, f, log_file=LOG_FILE):
        """
        Loads previously saved Pytorch model's state dictionary for inference.
        
        Args:
        
        **f**: the name of the file from where state dictionary will be loaded.
        
        **log_file**: name of the log file to save logs. If not specified, 
        logs are printed to stdout.
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
        """Saves full model (state dict, optimizer state, epochs, score, etc) 
        with the possibility to resume model training.
        
        Args:
        
        **f**: the name of the file where the model will be saved.
        
        **log_file**: name of the log file to save logs. If not specified, 
        logs are printed to stdout.
        """
        if log_file:
            print('Saving model...', end=' ', file=log_file)
            log_file.flush()
        torch.save(self, f, pickle_protocol=2)
        if log_file:
            print('done.', file=log_file)

    @staticmethod
    def load(f, device=None, log_file=LOG_FILE):
        """Loads previously saved full PyTorch model with the possibility 
        to resume training.
        
        Args:
        
        **f**: the name of the file from where the model will be loaded.
        
        **device**: device where the model will be loaded, 
        e.g. `torch.device('cuda:2')`. By default, the model is loaded to CPU.
        
        **log_file**: name of the log file to save logs. If not specified, 
        logs are printed to stdout.
        
        """
    
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
        """Creates model, criterion, optimizer and scheduler for training.
        Adam optimizer is used to train the model.
        
        Args:
        
        **cls**: classification model.
        
        **\*args**: args for the classification model.
        
        **lr**: learning rate for Adam optimizer.
        
        **\*\*kwargs**: keyword args for the classification model.
        
        """
    
        model = cls(*args, **kwargs)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        scheduler = None
        return model, criterion, optimizer, scheduler

    def adjust_model_for_tune(self, lr=.001, momentum=.9):
        """Ajusts model for post-train finetuning.
        Optimizer is changed to SGD to finetune the model.
        
        Args:
        
        **lr**: learning rate for the SGD optimizer. Default `lr=.001`
        
        **momentum**: momentum factor for the optimizer. Defaultm `momentum=.9`
        
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        scheduler = None
        return criterion, optimizer, scheduler
