# -*- coding: utf-8 -*-
# MorDL project: POS tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
import junky
from junky.dataset import BertDataset, CharDataset, DummyDataset, \
                          FrameDataset, LenDataset, TokenDataset, WordDataset
from mordl import WordEmbeddings
from mordl.base_tagger import BaseTagger
from mordl.lstm_tagger_model import LstmTaggerModel
from mordl.utils import LOG_FILE
import os
import sys


class PosTagger(BaseTagger):
    """"""

    def __init__(self):
        super().__init__()

    def predict(self):
        pass

    def evaluate(self):
        pass

    @staticmethod
    def prepare_corpus(corpus, tags_to_remove=None):
        return junky.extract_conllu_fields(
            junky.conllu_remove(corpus, remove=tags_to_remove),
            fields=['UPOS']
        )

    @staticmethod
    def create_dataset(sentences, word_emb_type=None, word_emb_path=None,
                       word_emb_model_device=None, word_transform_kwargs=None,
                       word_next_emb_params=None, with_chars=False,
                       labels=None):

        ds = FrameDataset()
        
        if word_emb_type is not None:
            x = WordEmbeddings.create_dataset(
                sentences, emb_type=word_emb_type, emb_path=word_emb_path,
                emb_model_device=word_emb_model_device,
                transform_kwargs=word_transform_kwargs,
                next_emb_params=word_next_emb_params
            )
            ds.add('x', x)
        else:
            ds.add('x', DummyDataset(data=sentences))
            ds.add('x_lens', LenDataset(data=sentences))

        if with_chars:
            x_ch = CharDataset(sentences,
                               unk_token='<UNK>', pad_token='<PAD>',
                               transform=True)
            ds.add('x_ch', x_ch, with_lens=False)
        else:
            ds.add('x_ch', DummyDataset(data=sentences))
            ds.add('x_ch_lens', DummyDataset(data=sentences))

        if labels:
            y = TokenDataset(labels, pad_token='<PAD>', transform=True,
                             keep_empty=False)
            ds.add('y', y, with_lens=False)

        return ds

    def train(self, model_file, model_config_file=True, device=None,
              epochs=sys.maxsize, min_epochs=0, bad_epochs=5,
              batch_size=32, control_metric='accuracy', max_grad_norm=None,
              tags_to_remove=None, word_emb_type=None, word_emb_path=None,
              word_emb_model_device=None, word_emb_tune_params=junky.kwargs(
                  name='bert-base-multilingual-cased', max_len=512,
                  epochs=4, batch_size=8
              ), word_transform_kwargs=None, word_next_emb_params=None,
              rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
              emb_out_dim=512, lstm_hidden_dim=256, lstm_layers=2, lstm_do=0,
              bn1=True, do1=.2, bn2=True, do2=.5, bn3=True, do3=.4, seed=None,
              log_file=LOG_FILE):

        assert self._train_corpus, 'ERROR: Train corpus is not loaded yet'

        if model_config_file is True and isinstance(model_file, str):
            pref, suff = os.path.splitext(model_file)
            model_config_file = pref + '.config' + suff

        def best_model_backup_method(model, model_score):
            if log_file:
                print('{}: new maximum score {:.8f}'
                          .format(iter_name, model_score),
                      end='')
            model.save_config(model_config_file, log_file=log_file)
            if model_config_file:
                model.save_state_dict(model_file, log_file=log_file)
            else:
                model.save(model_file, log_file=log_file)

        # 1. Prepare corpora
        train, train_labels = self.prepare_corpus(
            self._train_corpus, tags_to_remove=tags_to_remove
        )
        test, test_labels = self.prepare_corpus(
            self._train_corpus, tags_to_remove=tags_to_remove
        )

        # 2. Tune embeddings
        def tune_word_emb(emb_type, emb_path, emb_model_device=None,
                          emb_tune_params=None):
            if emb_tune_params is True:
                emb_tune_params = {}
            elif isinstance(emb_tune_params, str):
                emb_tune_params = {'model_name': emb_tune_params}
            if isinstance(emb_tune_params, dict):
                if emb_type == 'bert':
                    if 'test_data' not in emb_tune_params:
                        emb_tune_params['test_data'] = test, test_labels
                    emb_tune_params['save_to'] = emb_path
                    if emb_model_device and 'device' not in emb_tune_params:
                        emb_tune_params['device'] = emb_model_device
                    if 'seed' not in emb_tune_params:
                        emb_tune_params['seed'] = seed
                    if 'log_file' not in emb_tune_params:
                        emb_tune_params['log_file'] = log_file
                    emb_path = WordEmbeddings.bert_tune(
                        train, train_labels, **emb_tune_params
                    )['model_name']
                else:
                    raise ValueError("ERROR: tune method for '{}' embeddings "
                                         .format(emb_type)
                                   + 'is not implemented')
            return emb_path

        word_emb_path = tune_word_emb(
            word_emb_type, word_emb_path,
            emb_model_device=word_emb_model_device,
            emb_tune_params=word_emb_tune_params
        )
        if word_next_emb_params:
            if isinstance(word_next_emb_params, dict):
                word_next_emb_params = [word_next_emb_params]
            for emb_params in word_next_emb_params:
                tune_params = emb_params.get('emb_tune_params',
                              emb_params.get('word_emb_tune_params'))
                emb_params['emb_path'] = tune_word_emb(
                    emb_params.get('emb_type', emb_params['word_emb_type']),
                    emb_params.get('emb_path', emb_params['word_emb_path']),
                    emb_model_device=emb_params.get('emb_model_device',
                                     emb_params.get('word_emb_model_device'),
                                     word_emb_model_device),
                    emb_tune_params=\
                        emb_params.get('emb_tune_params',
                        emb_params.get('word_emb_tune_params'))
                )

        # 3. Create datasets
        ds_train = self.create_dataset(
            train, word_emb_type=word_emb_type, word_emb_path=word_emb_path,
            word_emb_model_device=word_emb_model_device,
            word_transform_kwargs=word_transform_kwargs,
            word_next_emb_params=word_next_emb_params,
            with_chars=rnn_emb_dim or cnn_emb_dim, labels=train_labels)
        ds_test = ds_train.clone(with_data=False)
        ds_test.transform(test,
                          names=[x for x in ds_test.list() if x != 'y'],
                          part_kwargs={})
        ds_test.transform(test_labels, names='y')

        if seed:
            junky.enforce_reproducibility(seed=seed)

        # 4. Create model
        model, criterion, optimizer, scheduler = \
            LstmTaggerModel.create_model_for_train(
                len(ds_train.get_dataset('y').transform_dict),
                tags_pad_idx=ds_train.get_dataset('y').pad,
                vec_emb_dim=ds_train.get_dataset('x').vec_size
                                if word_emb_type is not None else
                            None,
                alphabet_size=len(ds_train.get_dataset('x_ch').transform_dict)
                                  if rnn_emb_dim or cnn_emb_dim else
                              0,
                char_pad_idx=ds_train.get_dataset('x_ch').pad
                                 if rnn_emb_dim or cnn_emb_dim else
                             0,
                rnn_emb_dim=rnn_emb_dim,
                cnn_emb_dim=cnn_emb_dim, cnn_kernels=cnn_kernels,
                emb_out_dim=emb_out_dim, lstm_hidden_dim=lstm_hidden_dim,
                lstm_layers=lstm_layers, lstm_do=lstm_do,
                bn1=True, do1=.2, bn2=True, do2=.5, bn3=True, do3=.4
            )
        if device:
            model.to(device)
        if model_config_file:
            model.save_config(model_config_file, log_file=log_file)

        # 5. Train model
        res_ = junky.train(
            device, None, model, criterion, optimizer, scheduler,
            None, '', datasets=(ds_train, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric='accuracy',
            max_grad_norm=max_grad_norm,
            with_progress=log_file is not None, log_file=log_file
        )
        if model_config_file:
            model.load_state_dict(model_file, log_file=log_file)
        else:
            del model
            model = LstmTaggerModel.load(model_file, log_file=log_file)
        best_epoch, best_score = res_['best_epoch'], res_['best_score']
        res = {x: y[:best_epoch + 1]
                   for x, y in res_.items()
                       if x not in ['best_epoch', 'best_score']}

        # 6. Tune model
        criterion, optimizer, scheduler = model.adjust_model_for_tune()
        res_= junky.train(
            device, model, criterion, optimizer, scheduler,
            lambda x, y: x.save_state_dict(model_file)
                             if model_config_file else
                         x.save(model_file),
            '', datasets=(ds_train, ds_test),
            epochs=epochs, min_epochs=min_epochs, bad_epochs=bad_epochs,
            batch_size=batch_size, control_metric='accuracy',
            max_grad_norm=max_grad_norm, best_score=best_score,
            with_progress=log_file is not None, log_file=log_file
        )
        best_epoch = res_['best_epoch']
        res.update({x: y[:best_epoch + 1]
                        for x, y in res_.items()
                            if x not in ['best_epoch', 'best_score']})

        return res
