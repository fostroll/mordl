# -*- coding: utf-8 -*-
# MorDL project: Word embeddings helper
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides a class for loading and applying pretrained embeddings to
input sentences and creating datasets and collated dataloaders.
"""
import logging
import sys
#logging.basicConfig(level=logging.ERROR)
#if not sys.warnoptions:
#    import warnings
#    warnings.simplefilter('ignore')
#    os.environ['PYTHONWARNINGS'] = 'ignore'
from collections.abc import Iterable
from copy import deepcopy
from corpuscula.corpus_utils import get_root_dir
#from gensim.models.fasttext import load_facebook_model
from gensim.models.fasttext import load_facebook_vectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
from gensim.models.keyedvectors import FastTextKeyedVectors
import json
import junky
from junky.dataset import BaseDataset, BertDataset, LenDataset, \
                          WordCatDataset, WordDataset
from junky.trainer import Trainer, TrainerConfig
from mordl.defaults import BATCH_SIZE, CONFIG_ATTR, CONFIG_EXT, LOG_FILE
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            f1_score, precision_score, recall_score
from tempfile import mkstemp
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoConfig, AutoModel, AutoTokenizer, \
                         get_linear_schedule_with_warmup
#BertConfig, BertForSequenceClassification, \
#                         BertForTokenClassification, BertTokenizer, \

_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS = junky.kwargs(
    max_len=0, batch_size=BATCH_SIZE, hidden_ids=10,
    aggregate_hiddens_op='cat', aggregate_subtokens_op='absmax',
    to=junky.CPU, loglevel=1
)
_DEFAULT_DATASET_TRANSFORM_KWARGS = junky.kwargs(
    check_lower=True
)
_MAX_BAD_EPOCHS = 0

# to suppress transformers' warnings
#logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
logging.getLogger('transformers').setLevel(logging.ERROR)
#logging.getLogger('pytorch_pretrained_bert.tokenization').setLevel(logging.ERROR)


class WordEmbeddings:
    """
    The class that handles loading various pretrained word embeddings.
    """

    @staticmethod
    def full_tune(model, model_save_as, model_save_method,
                  datasets, sents_data, device=None,
                  control_metric='accuracy', best_score=None,
                  # word_emb_tune_params ##################
                  save_as=None, epochs=3, batch_size=8,
                  lr=2e-5, betas=(0.9, 0.999), eps=1e-8,
                  weight_decay=.01, amsgrad=False,
                  num_warmup_steps=3, max_grad_norm=1.,
                  #########################################
                  transform_kwargs=None,
                      # BertDataset.transform() # params:
                      # {'max_len': 0, 'batch_size': 64, 'hidden_ids': '10',
                      #  'aggregate_hiddens_op': 'cat',
                      #  'aggregate_subtokens_op': 'absmax', 'to': junky.CPU,
                      #  'loglevel': 1}
                      # NB: transform_kwargs['batch_size'] is ignored and
                      #     replaced with the **batch_size** param.
                  seed=None, log_file=LOG_FILE):
        """The method for finetuning base BERT model on custom data.

        Args:

        **model**: the head model using after BERT.

        **model_save_as** (`str`): the path where the head model will be
        stored.

        **model_save_method** (`callable`): the method using to save the model
        head. The signature: `model_save_method(model_save_as)`.

        **datasets** (`tuple(junky.dataset.FrameDataset,
        junky.dataset.FrameDataset | None) | junky.dataset.FrameDataset`): the
        train and (possibly) test datasets that are valid for the **model**
        training.

        **sents_data** (`tuple(list([str]), list([str]) | None) |
        list([str])): the sequence of already tokenized sentences that
        corresponds to the data in **datasets**. It is used to train the
        embedding model inside the `'x'` nested dataset of the train part of
        the **datasets**.

        **device**: if not `None`, the full model will be transfered to the
        specified device.

        **control_metric** (of `str` type; default is `'accuracy'`): the
        metric to control the model performance in the validation time. The
        vaues allowed are: `'loss'`, `'accuracy'`, `'precision'`, `'recall'`,
        `'f1'`.

        **best_score** (`float`; default is `None`): the starting point to
        compare the calculating control metric with.

        **save_as** (`str`): the path where the finetuned BERT model will be
        saved. This parameter must not be `None`.

        **epochs** (`int`; default is `3`): the number of finetuning epochs.

        **batch_size** (`int`; default is `8`): number of sentences per batch.

        **lr** (default is `5e-5`), **betas** (default is `(0.9, 0.999)`),
        **eps** (default is `1e-8`), **weight_decay** (default is `0.01`),
        **amsgrad** (default is `False`): params for *AdamW* optimizer.

        **num_warmup_steps** (`int` | `float`): the number of warmup steps for
        the scheduler.

        **max_grad_norm** (`float`; default is `None`): the gradient clipping
        parameter.

        **transform_kwargs** (`dict`; default is `None`): keyword arguments
        for the `.transform()` method of `junky.datasets.BertDataset` if you
        want to learn allowed values for the parameter. If `None`, the
        `.transform()` method use its defaults.

        **seed** (`int`; default is `None`): init value for the random number
        generator if you need reproducibility. Note that each stage will have
        its own seed value, and the **seed** param is used to calculate these
        values.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns the finetune statistics.
        """
        assert control_metric in ['accuracy', 'f1', 'loss'], \
           f'ERROR: Unknown control_metric "{control_metric}" ' \
            "(only 'accuracy', 'f1' and 'loss' are available)."
        assert save_as, 'ERROR: Undefined `save_as` param.'

        train_ds, test_ds = datasets if isinstance(datasets, tuple) else \
                            (datasets, None)
        train_sents, test_sents = sents_data \
                                      if isinstance(sents_data, tuple) else \
                                  (sents_data, None)
        assert (test_ds and test_sents) or not (test_ds or test_sents), \
            'ERROR: given {} without {}.' \
                .format(*(('test dataset', 'test sentences') if test_ds else
                          ('test sentences', 'test dataset')))
        assert len(train_ds) == len(train_sents), \
            'ERROR: train dataset and train corpus have different sizes.'
        assert not test_ds or len(test_ds) == len(test_sents), \
            'ERROR: test dataset and test corpus have different sizes.'

        kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
        if transform_kwargs:
            kwargs.update(transform_kwargs)
        transform_kwargs = kwargs

        if seed:
            junky.enforce_reproducibility(seed)

        class WordEmbeddingsModel(nn.Module):

            def __init__(self, bert_ds, model_head):
                super().__init__()

                self.ds = bert_ds
                self.emb_model = bert_ds.model
                self.tokenizer = bert_ds.tokenizer
                self.model_head = model_head

            def save_pretrained(self, paths):
                self.tokenizer.save_pretrained(paths[0])
                self.emb_model.save_pretrained(paths[0])
                model_save_method(paths[1])

            def forward(self, sentences, *args, labels=None):
                x = self.ds.transform(
                    sentences, max_len=transform_kwargs['max_len'],
                    batch_size=batch_size,
                    hidden_ids=transform_kwargs['hidden_ids'],
                    aggregate_hiddens_op=
                        transform_kwargs['aggregate_hiddens_op'],
                    aggregate_subtokens_op=
                        transform_kwargs['aggregate_subtokens_op'],
                    with_grad=self.training, save=False, loglevel=0
                )
                x, x_lens = self.ds._collate(x, with_lens=True)
                return self.model_head(x, x_lens, *args, labels=labels)

        train_ds_x = BaseDataset([list(x) for x in train_sents])
        train_ds_len = LenDataset(train_ds_x)
        if test_sents:
            test_ds_x = BaseDataset([list(x) for x in test_sents])
            test_ds_len = LenDataset(test_ds_x)

        train_ds_bert = train_ds.datasets['x']
        test_ds_bert = test_ds.datasets['x']
        bert_ds = train_ds_bert[0].clone(with_data=False)
        if test_sents:
            train_ds_bert = (bert_ds, *train_ds_bert[1:])
            test_ds_bert = (bert_ds.clone(), *test_ds_bert[1:])

        train_ds.datasets['x'] = (train_ds_x, 1, {})
        train_ds.add('len', train_ds_len)
        if test_sents:
            test_ds.datasets['x'] = (test_ds_x, 1, {})
            test_ds.add('len', test_ds_len)

        train_dl = train_ds.create_loader(
            batch_size=batch_size, shuffle=True, num_workers=0
        )
        if test_sents:
            test_dl = test_ds.create_loader(
            batch_size=transform_kwargs['batch_size'], shuffle=False,
            num_workers=0
        )

        full_model = WordEmbeddingsModel(bert_ds, model)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(full_model.emb_model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                                  if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .01},
                {'params': [p for n, p in param_optimizer
                                  if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .0}
            ]
        else:
            param_optimizer = \
                list(full_model.emb_model.classifier.named_parameters())
            optimizer_grouped_parameters = \
                [{'params': [p for n, p in param_optimizer]}]

        optimizer_grouped_parameters.append(
            {'params': list(full_model.model_head.parameters())}
        )

        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, betas=betas,
                          eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

        max_grad_norm = max_grad_norm

        # Total number of training steps is
        # number of batches * number of epochs
        total_steps = len(train_dl) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
        # scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps if num_warmup_steps >= 1 else
                             int(num_warmup_steps * total_steps),
            num_training_steps=total_steps
        )

        trainer_config = TrainerConfig(
            (save_as, model_save_as), max_epochs=epochs,
            batch_lens_idx=-1, batch_labels_idx=-2,
            model_args=list(range(len(next(iter(train_dl))) - 2)),
            model_kwargs={'labels': -2},
            output_logits_idx=0, output_loss_idx=1,
            max_grad_norm=max_grad_norm, optimizer=optimizer,
            scheduler=scheduler, postprocess_method='strip_mask',
            save_ckpt_method=
                lambda model, paths: full_model.save_pretrained(paths)
        )
        trainer = Trainer(trainer_config, full_model, train_dl,
                          test_dataloader=test_dl if test_sents else None,
                          device=device)

        try:
            res = trainer.train(best_score=best_score)
        except RuntimeError as e:
            if e.args and e.args[0].startswith('CUDA out of memory'):
                e = RuntimeError(
                    e.args[0] + '. To avoid this, consider to decrease '
                                'batch_size or max_len value(s)',
                    *e.args[1:]
                )
            raise e

        train_ds.remove('len')
        test_ds.remove('len')
        return res

    @staticmethod
    def bert_tune(train_sentences, train_labels, test_data=None,
                  model_name='bert-base-multilingual-cased', device=None,
                  save_to=None, max_len=512, epochs=3, batch_size=8,
                  special_tokens=None, control_metric='accuracy', seed=None,
                  log_file=LOG_FILE):
        """Method for finetuning base BERT model on custom data.

        Args:

        **train_sentences**: sequence of already tokenized sentences (of the
        `list([str])` format) that will be used to train the model.

        **train_labels**: list of labels (of `str`) that will be used to
        train the model.

        **test_data** (`tuple(<sentences>, <labels>)`): development test set to
        validate the model during training.

        **model_name** (`str`): pre-trained BERT model name or path to the
        model. Default `model_name='bert-base-multilingual-cased'`.

        **device**: device for the BERT model. Default device is CPU.

        **save_to** (`str`): a path where the finetuned BERT model will be
        saved. If not specified, it will be generated based on **model_name**
        and the model tune parameters. If **save_to** is ended on the `'_'`
        symbol, it will be treated as a prefix for the autogenerated name.

        **max_len** (`int`): maximal input sequence length. By default,
        restricted by BERT's positional embeddings, `max_len=512`. Don't make
        in higher.

        **epochs** (`int`): maximal number of finetuning epochs. Default
        `epochs=3`.

        **batch_size** (`int`): number of sentences per batch. Default
        `batch_size=8`.

        **special_tokens** (`str`|`list(<str>)`): additional special tokens
        for BERT tokenizer.

        **control_metric** (`str`): metric to control training. Allowed values
        are: 'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

        **seed** (`int`): random seed.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns the finetune statistics.
        """
        assert control_metric in ['accuracy', 'f1', 'loss'], \
               "ERROR: Unknown control_metric '{}' ".format(control_metric) \
             + "(only 'accuracy', 'f1' and 'loss' are available)"

        prefix = ''
        if save_to and save_to.endswith('_'):
            prefix, save_to = save_to, prefix
        if not save_to:
            save_to = '{}{}_len{}_ep{}_bat{}'.format(prefix, model_name,
                                                     max_len, epochs,
                                                     batch_size)
            if seed is not None:
                save_to += '_seed{}'.format(seed)

        if log_file:
            print("BERT MODEL TUNING '{}'. The result's model name will be '{}'"
                      .format(model_name, save_to),
                  file=log_file)

        test_sentences, test_labels = test_data if test_data else ([], [])

        use_seq_labeling = train_labels and not isinstance(train_labels[0],
                                                           tuple)

        if seed:
            junky.enforce_reproducibility(seed)

        def seq2ix(seq, extra_labels=None):
            """Method to create sequence-to-index dictionary.

            Args:

            **seq**: sequence of tokenized sentences.

            **extra_labels**: any additional tokens or labels to add
            to the dictionary.
            """
            labels = set(x for x in seq) if use_seq_labeling else \
                     set(x for x in seq for x in x)
            seq2ix = {x: i for i, x in enumerate(sorted(labels))}
            if extra_labels:
                for tag in extra_labels:
                    seq2ix[tag] = len(seq2ix)
            return seq2ix

        cls_label, sep_label, pad_label = '<CLS>', '<SEP>', '<PAD>'

        t2y = seq2ix(train_labels,
                     extra_labels=None if use_seq_labeling else
                                  [cls_label, sep_label, pad_label])
        y2t = {v:k for k, v in t2y.items()}

        if not use_seq_labeling:
            cls_y, sep_y, pad_y = \
                [t2y[x] for x in [cls_label, sep_label, pad_label]]
        # undefined elsewise

        if log_file:
            print('Loading tokenizer...', file=log_file)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        if log_file:
            print('Tokenizer is loaded. Vocab size:', tokenizer.vocab_size,
                  file=log_file)

        def prepare_corpus(sentences, labels, max_len=None):
            """Method to run input sentences through tokenization and
            apply overlays if input sentence is longer than the specified
            `max_len`.
            """

            def tokenize_and_preserve_labels(sentence, text_labels):
                """Tokenizes input using BertTokenizer from tranformers library.
                Each token's labels are broadcasted to all wordpieces of that
                token.

                Args:

                **sentence**: input sentence.

                **text_labels**: labels for each token from the input sentence.
                """
                tokenized_sentence = []
                labels = []

                for word, label in zip(sentence, [[]] * len(sentence)
                                                     if use_seq_labeling else
                                                 text_labels):
                    # Tokenize the word and count
                    # of subwords the word is broken into
                    tokenized_word = tokenizer.tokenize(word)
                    n_subwords = len(tokenized_word)

                    # Add the tokenized word to the final tokenized word list
                    tokenized_sentence.extend(tokenized_word)

                    if not use_seq_labeling:
                        # Add the same label to the new list
                        # of labels `n_subwords` times
                        labels.extend([t2y[label]] * n_subwords)

                return tokenized_sentence, t2y[text_labels] \
                                               if use_seq_labeling else \
                                           labels

            def apply_max_len(sents, labs, max_len):
                """Applies max_len restriction to split input sentences
                and labels with overlays.

                Args:

                **sents**: input sentences.

                **labs**: input labels.

                **max_len**: maximum input sequence length.
                """
                if max_len:
                    max_len -= 2
                    sents_, labs_ = [], []
                    for sent, lab in zip(sents, labs):
                        sent_len = len(sent)
                        cnt = sent_len // max_len + 1
                        if cnt == 1:
                            sents_.append(sent)
                            labs_.append(lab)
                        else:
                            step = (sent_len - max_len) // (cnt - 1)
                            for i in range(cnt):
                                start = i * step
                                for i in range(start, sent_len):
                                    if sent[i][:2] != '##':
                                        start = i
                                        break
                                end = min(start + max_len, sent_len)
                                if end != sent_len:
                                    for i in reversed(range(end)):
                                        if sent[i][:2] != '##':
                                            end = i
                                            break
                                sents_.append(sent[start:end])
                                labs_.append(lab if use_seq_labeling else
                                             lab[start:end])
                    sents, labs = sents_, labs_
                return sents, labs

            return apply_max_len(*zip(*[
                tokenize_and_preserve_labels(sent, labs)
                    for sent, labs in zip(sentences, labels)
            ]), max_len)

        if log_file:
            print('Corpora processing...', end=' ', file=log_file)
            log_file.flush()
        train_sentences_, train_labels_ = \
            prepare_corpus(train_sentences, train_labels, max_len)
        test_sentences_, test_labels_ = \
            prepare_corpus(test_sentences, test_labels, max_len)
        if log_file:
            print('done.', file=log_file)

        def collate(batch):
            """Encodes and pads input batch, adding special tokens
            (`[CLS]`, `[SEP]`) and creating attention mask for the batch.
            Output input indexes, attention masks, label indexes and
            sequence lengths.

            Args:

            **batch**: input batch - tokenized sentences and aligned labels.
            """
            sents, labels = zip(*batch)
            max_len_ = max(len(x) for x in sents)
            encoded_sents = [
                tokenizer.encode_plus(text=sent,
                                      add_special_tokens=True,
                                      max_length=max_len_ + 2,
                                      #truncation=True,
                                      #pad_to_max_length=True,
                                      padding='max_length',
                                      return_tensors='pt',
                                      return_attention_mask=True,
                                      return_overflowing_tokens=False)
                    for sent in sents
            ]
            input_ids, attention_masks = zip(*[
                (x['input_ids'], x['attention_mask'])
                    for x in encoded_sents
            ])
            input_ids = torch.cat(input_ids, dim=0)
            attention_masks = torch.cat(attention_masks, dim=0)
            lens = attention_masks.sum(dim=1) - 2
            output_ids = torch.tensor(labels if use_seq_labeling else \
                                      [[cls_y] + x + [sep_y]
                                               + [pad_y] * (max_len_ - len(x))
                                           for x in labels])
            return input_ids, attention_masks, output_ids, lens

        class SubwordDataset(Dataset):
            """Creates subword dataset from sentences tokenized with
            `transformers.BertTokenizer`.
            """
            def __init__(self, x_data, y_data):
                super().__init__()
                self.x_data = x_data
                self.y_data = y_data

            def __len__(self):
                return len(self.x_data)

            def __getitem__(self, idx):
                return self.x_data[idx], self.y_data[idx]

        train_dataset = SubwordDataset(train_sentences_, train_labels_)
        test_dataset = SubwordDataset(test_sentences_, test_labels_)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=0, shuffle=True,
                                  collate_fn=collate)
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                 num_workers=0, shuffle=False,
                                 collate_fn=collate)

        if log_file:
            print("Loading model '{}'...".format(model_name), end=' ',
                  file=log_file)
            log_file.flush()
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=len(t2y),
            output_attentions = False, output_hidden_states = False
        ) if use_seq_labeling else \
        BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(t2y),
            output_attentions = False, output_hidden_states = False
        )
        if special_tokens:
            tokenizer.add_tokens(special_tokens, special_tokens=True)
            model.resize_token_embeddings(len(tokenizer))
        if device:
            model.to(device)
        if log_file:
            print('done.', file=log_file)

        FULL_FINETUNING = True
        if FULL_FINETUNING:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer
                                  if not any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .01},
                {'params': [p for n, p in param_optimizer
                                  if any(nd in n for nd in no_decay)],
                 'weight_decay_rate': .0}
            ]
        else:
            param_optimizer = list(model.classifier.named_parameters())
            optimizer_grouped_parameters = \
                [{'params': [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

        max_grad_norm = 1.

        # Total number of training steps is
        # number of batches * number of epochs
        total_steps = len(train_loader) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
        # scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        def save_finetuned_bert(model, output_dir):
            """Saves finetuned BERT model to the specified output directory.

            Args:

            **model**: finetuned BERT model.

            **output_dir**: output directory to save the finetuned model.
            """
            # Saving best-practices: if you use defaults names for the model,
            # you can reload it using from_pretrained()

            # Create output directory if needed
            if not os.path.exists(save_to):
                os.makedirs(save_to)

            if log_file:
                print('Saving model to {}'.format(save_to), file=log_file)

            # Save a trained model, configuration and tokenizer
            # using `save_pretrained()`. They can then be reloaded
            # using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else \
                            model
                # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Good practice: save your training arguments together
            # with the trained model:
            # torch.save(args, os.path.join(output_dir, 'training_args.bin'))

        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        best_score = float('-inf')
        best_test_golds, best_test_preds = [], []

        bad_epochs = 0

        junky.clear_tqdm()

        # Store the average loss after each epoch so we can plot them
        loss_values, test_loss_values = [], []

        try:
            for epoch in range(1, epochs + 1):
                # ========================================
                #               Training
                # ========================================
                # Perform one full pass over the training set.

                # Reset the total loss for this epoch.
                total_loss = 0

                progress_bar = tqdm(desc='Epoch {}'.format(epoch),
                                    total=len(train_loader.dataset),
                                    file=log_file) \
                                   if log_file else \
                               None

                # Put the model into training mode.
                model.train()
                # Training loop
                t, n_update = time.time(), 0
                for step, batch in enumerate(train_loader):
                    if device:
                        # move batch to the specified device
                        batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels, lens = batch
                    # Always clear any previously calculated gradients
                    #before performing a backward pass.
                    model.zero_grad()
                    # forward pass
                    # This will return the loss (rather than the model output)
                    # because we provide labels
                    outputs = model(
                        b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels
                    )
                    # get the loss
                    loss = outputs[0]
                    # Perform a backward pass to calculate the gradients
                    loss.backward()
                    # track train loss
                    total_loss += loss.item()
                    # Clip the norm of the gradient. It prevents the
                    # "exploding gradients" problem
                    torch.nn.utils.clip_grad_norm_(
                        parameters=model.parameters(),
                        max_norm=max_grad_norm
                    )
                    # update parameters
                    optimizer.step()
                    # Update the learning rate
                    scheduler.step()

                    if log_file:
                        t_ = time.time()
                        n_update += b_input_ids.shape[0]
                        if t_ - t >= 2:
                            t = t_
                            progress_bar.set_postfix(train_loss=loss.item())
                            progress_bar.update(n_update)
                            n_update = 0

                if log_file:
                    if n_update:
                        progress_bar.update(n_update)
                    progress_bar.close()

                # Calculate the average loss over the training data
                avg_train_loss = total_loss / len(train_loader)
                if log_file:
                    print('Average train loss: {}'.format(avg_train_loss),
                          file=log_file)

                # Store the loss value for plotting the learning curve
                loss_values.append(avg_train_loss)

                # ========================================
                #               Validation
                # ========================================
                # After the completion of each training epoch, measure
                # our performance on our validation set

                # Put the model into evaluation mode
                if test_data:
                    model.eval()
                    # Reset the validation loss for this epoch
                    eval_loss, eval_accuracy = 0, 0
                    nb_eval_steps, nb_eval_examples = 0, 0
                    gold_labels, pred_labels = [], []
                    for batch in test_loader:
                        if device:
                            batch = tuple(t.to(device) for t in batch)
                        b_input_ids, b_input_mask, b_labels, lens = batch

                        # Telling the model not to compute or store gradients,
                        # saving memory and speeding up validation
                        with torch.no_grad():
                            # Forward pass, calculate logit predictions.
                            # This will return the pred_ids rather than the
                            # loss because we don't provide labels
                            outputs = model(
                                b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels
                            )
                        # Move logits and labels to CPU
                        pred_ids = outputs[1].detach().cpu().numpy()
                        gold_ids = b_labels.to(junky.CPU).numpy()

                        # Calculate the accuracy for this batch of test
                        # sentences
                        eval_loss += outputs[0].mean().item()
                        pred_labels.extend(
                            np.argmax(pred_ids, axis=1) if use_seq_labeling else
                            [list(x)[1:y + 1]
                                 for x, y in zip(np.argmax(pred_ids, axis=2),
                                                           lens)]
                        )
                        gold_labels.extend(
                            gold_ids if use_seq_labeling else
                            [x[1:y + 1].tolist() for x, y in zip(gold_ids,
                                                                 lens)]
                        )

                        nb_eval_examples += b_input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    test_loss_values.append(eval_loss)

                    if use_seq_labeling:
                        gold_labels = [x for x in gold_labels]
                        pred_labels = [x for x in pred_labels]
                    else:
                        gold_labels = [x for x in gold_labels for x in x]
                        pred_labels = [x for x in pred_labels for x in x]

                    accuracy = accuracy_score(gold_labels, pred_labels)
                    precision = precision_score(gold_labels, pred_labels,
                                                average='macro')
                    recall = recall_score(gold_labels, pred_labels,
                                          average='macro')
                    f1 = f1_score(gold_labels, pred_labels, average='macro')

                    accuracies.append(accuracy)
                    precisions.append(precision)
                    recalls.append(recall)
                    f1s.append(f1)

                    score = -eval_loss if control_metric == 'loss' else \
                            accuracy if control_metric == 'accuracy' else \
                            f1 if control_metric == 'f1' else \
                            None

                    if log_file:
                        print('Average test loss: {}'.format(eval_loss),
                              file=log_file)
                        print('Dev: accuracy = {:.8f}'.format(accuracy),
                              file=log_file)
                        print('Dev: precision = {:.8f}'.format(precision),
                              file=log_file)
                        print('Dev: recall = {:.8f}'.format(recall),
                              file=log_file)
                        print('Dev: f1_score = {:.8f}'.format(f1),
                              file=log_file)
                        if not use_seq_labeling:
                            print('NB: Scores may be high because of labels '
                                  'stretching', file=log_file)

                    if score > best_score:
                        best_score = score
                        best_test_golds, best_test_preds = \
                            gold_labels[:], pred_labels[:]

                        save_finetuned_bert(model, output_dir=save_to)
                        bad_epochs = 0

                    else:
                        bad_epochs += 1
                        if log_file:
                            print('BAD EPOCHS:', bad_epochs, file=log_file)
                        if _MAX_BAD_EPOCHS and bad_epochs >= _MAX_BAD_EPOCHS:
                            if log_file:
                                print('Maximum bad epochs exceeded. '
                                      'Process was stopped.', file=log_file)
                            break

                if epoch == epochs:
                    print('\nMaximum epochs exceeded. ' \
                          'Process has been stopped.')

                if log_file:
                    log_file.flush()

        except RuntimeError as e:
            if e.args and e.args[0].startswith('CUDA out of memory'):
                e = RuntimeError(
                    e.args[0] + '. To avoid this, consider to decrease '
                                'batch_size or max_len value(s).',
                    *e.args[1:]
                )
            raise e

        del model

        return {'model_name': save_to,
                'best_score': best_score,
                'best_test_golds': best_test_golds,
                'best_test_preds': best_test_preds,
                'loss_values': loss_values,
                'test_loss_values': test_loss_values,
                'accuracies': accuracies,
                'precisions': precisions,
                'recalls': recalls,
                'f1s': f1s}

    @staticmethod
    def load(emb_type, emb_path, emb_model_device=None, embs=None):
        """Method to load pretrained embeddings model from **emb_path**.

        Args:

        **emb_type**: (`str`) one of the supported embeddings types. Allowed
        values: 'bert' for *BERT*, 'ft' for *FastText*, 'glove' for *Glove*,
        'w2v' for *Word2vec*.

        **emb_path**: path to the embeddings file.

        **emb_model_device**: relevant with `emb_type='bert'`. The device
        where the BERT model will be loaded to.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If **emb_path** is in
        **embs**, the method just return the corresponding model.

        Returns the embeddings model.
        """
        if embs and emb_path in embs:
            model = embs[emb_path]
        else:
            emb_type = emb_type.lower()
            if emb_type == 'bert':
                tokenizer = AutoTokenizer.from_pretrained(
                    emb_path, do_lower_case=False
                )
                config = AutoConfig.from_pretrained(
                    emb_path, output_hidden_states=True,
                    output_attentions=False
                )
                model = AutoModel.from_pretrained(emb_path, config=config)
                if emb_model_device:
                    model.to(emb_model_device)
                model.eval()
                model = model, tokenizer

            elif emb_type == 'glove':
                try:
                    model = KeyedVectors.load_word2vec_format(emb_path,
                                                              binary=False)
                except ValueError:
                    fn = os.path.basename(emb_path)
                    pref, suff = os.path.splitext(fn)
                    dn = get_root_dir()
                    f, fn = mkstemp(suffix=suff, prefix=pref, dir=nd)
                    f.close()
                    glove2word2vec(emb_path, fn)
                    model = KeyedVectors.load_word2vec_format(fn,
                                                              binary=False)
                    os.remove(fn)
                except UnicodeDecodeError:
                    model = KeyedVectors.load(emb_path)
                model = {x: model.vectors[y.index]
                             for x, y in model.vocab.items()}

            elif emb_type in ['ft', 'fasttext']:
                try:
                    #model = load_facebook_model(emb_path).wv
                    model = load_facebook_vectors(emb_path)
                except NotImplementedError:
                    model = KeyedVectors.load(emb_path)
                    if not isinstance(model, FastTextKeyedVectors):
                        raise ValueError('ERROR: Unable to load '
                                         'Word2vec vectors as FastText')

            elif emb_type in ['w2v', 'word2vec']:
                try:
                    model = KeyedVectors.load_word2vec_format(emb_path,
                                                              binary=False)
                except UnicodeDecodeError:
                    try:
                        model = KeyedVectors.load_word2vec_format(emb_path,
                                                                  binary=True)
                    except UnicodeDecodeError:
                        try:
                            model = load_facebook_model(emb_path).wv
                        except NotImplementedError:
                            model = KeyedVectors.load(emb_path)
                model = {x: model.vectors[y.index]
                             for x, y in model.vocab.items()}

            else:
                raise ValueError('ERROR: Unknown emb_type. Allowed values: '
                                 "'bert' for BERT, 'glove' for GloVe, "
                                 "'ft' for fastText, 'w2v' for Word2vec")

            if embs is not None:
                embs[emb_path] = model
        return model

    @classmethod
    def create_dataset(cls, sentences, emb_type='ft', emb_path=None,
                       emb_model_device=None, transform_kwargs=None,
                       next_emb_params=None, embs=None, loglevel=2):
        """Creates dataset with embedded sequences.

        Args:

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format) that will be used for initial transform of
        the created dataset.

        **emb_type**: one of ('bert'|'glove'|'ft'|'w2v') embedding types.

        **emb_path** (`str`): path to word embeddings storage.

        **emb_model_device**: the torch device where the model of word
        embeddings are placed. Relevant only with embedding types, models of
        which use devices (currently, only 'bert').

        **transform_kwargs** (`dict`): keyword arguments for `.transform()`
        method of the dataset created for sentences to word embeddings
        conversion. See the `.transform()` method of
        `junky.datasets.BertDataset` for the the description of the
        parameters.

        **next_emb_params**: if you want to use several different embedding
        models at once, pass parameters of the additional model as a
        dictionary with keys `(emb_path, emb_model_device, transform_kwargs)`;
        or a list of such dictionaries if you need more than one additional
        models.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If **emb_path** is in
        **embs**, the method don't load the corresponding model and use
        already loaded one.

        **loglevel**: param for dataset's `.transform()`. Relevant with
        `emb_type='bert'`.

        Returns a loaded dataset.
        """
        emb_params = \
            [(emb_type, emb_path, emb_model_device, transform_kwargs)]
        if next_emb_params:
            if isinstance(next_emb_params, dict) \
            or isinstance(next_emb_params[0], str):
                next_emb_params = [next_emb_params]
            for emb_params_ in next_emb_params:
                if isinstance(emb_params_, dict):
                    emb_params.append(
                        (emb_params_.get('emb_type',
                         emb_params_.get('word_emb_type',
                         emb_params_['emb_type'])),  # for correct value name
                                                     # in error message
                         emb_params_.get('emb_path',
                         emb_params_.get('word_emb_path',
                         emb_params_['emb_path'])),  # for correct value name
                                                     # in error message
                         emb_params_.get('emb_model_device',
                         emb_params_.get('word_emb_model_device')),

                         emb_params_.get('transform_kwargs',
                         emb_params_.get('word_transform_kwargs')))
                     )
                else:
                    emb_params.append(emb_params_)

        ds, config = [], []
        for emb_params_ in emb_params:
            emb_type, emb_path = emb_params_[:2]
            emb_model_device = emb_params_[2] if len(emb_params_) > 2 else \
                               None
            transform_kwargs = emb_params_[3] if len(emb_params_) > 3 else \
                               None
            if isinstance(emb_model_device, dict) or (
                transform_kwargs and not isinstance(transform_kwargs, dict)
            ):
                emb_model_device, transform_kwargs = \
                    transform_kwargs, emb_model_device

            config.append(junky.kwargs_nonempty(
                emb_type=emb_type, emb_path=emb_path,
                emb_model_device=str(emb_model_device),
                transform_kwargs=transform_kwargs
            ))

            if transform_kwargs is None:
                transform_kwargs = {}

            emb_model = cls.load(emb_type, emb_path,
                                 emb_model_device=emb_model_device, embs=embs)

            if emb_type == 'bert':
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                if loglevel is not None:
                    kwargs['loglevel'] = loglevel
                transform_kwargs = kwargs
                model, tokenizer = emb_model
                x = BertDataset(model, tokenizer)
            else:
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
                x = WordDataset(
                    emb_model=emb_model,
                    vec_size=len(next(iter(emb_model.values())))
                                 if isinstance(emb_model, dict) else
                             emb_model.vector_size,
                    unk_token='<UNK>', pad_token='<PAD>'
                )
            junky.clear_tqdm()
            x.transform(sentences, **transform_kwargs)
            ds.append(x)

        if len(ds) == 1:
            ds, config = ds[0], config[0]
        else:
            ds_ = WordCatDataset()
            for i, x in enumerate(ds):
                ds_.add('x_{}'.format(i), x)
            ds = ds_

        if hasattr(ds, CONFIG_ATTR):
            raise AttributeError(
                'ERROR: {} class has unexpected attribute `{}`'
                    .format(ds.__class__, CONFIG_ATTR)
            )
        setattr(ds, CONFIG_ATTR, config)

        return ds

    @classmethod
    def transform(cls, ds, sentences, transform_kwargs=None,
                  batch_size=BATCH_SIZE, loglevel=1):
        """Converts **sentences** of tokens to the sequences of the
        corresponding indices.

        Args:

        **ds**: a dataset to transform **sentences**.

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format).

        **transform_kwargs**: keyword arguments for `.transform` function.

        **batch_size**: number of sentences per batch. If not `None`,
        overwrites `batch_size` parameter from `transform_kwargs`.
        By default, `batch size=64`.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns `True` if **ds** is of appropriate type and the transforming
        have been done). Otherwise, returns `False`, that means that you have
        to transform data by it's own method.
        """
        res = False
        config = getattr(ds, CONFIG_ATTR, None)
        if config:
            if isinstance(ds, WordCatDataset):
                for name, cfg in zip(ds.list(), config):
                    res = res \
                      and cls.transform_dataset(ds.get_dataset(name),
                                                sentences)
                    if not res:
                        break
            else:
                kwargs = config.get('transform_kwargs', {})
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
        for _ in range(1):
            if isinstance(ds, BertDataset):
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if batch_size:
                    kwargs['batch_size'] = batch_size
                if loglevel is not None:
                    kwargs['loglevel'] = loglevel
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            elif isinstance(ds, WordDataset):
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            else:
                break
            junky.clear_tqdm()
            ds.transform(sentences, **transform_kwargs)
            res = True
        return res

    @classmethod
    def transform_collate(cls, ds, sentences, transform_kwargs=None,
                          batch_size=BATCH_SIZE, loglevel=1):
        """Sequentially makes batches from `sentences`.

        Args:

        **ds**: a dataset to transform **sentences**.

        **sentences**: an input sequence of already tokenized sentences (of
        the `list([str])` format).

        **transform_kwargs**: keyword arguments for the `.transform` function
        of the **ds**.

        **batch_size**: number of sentences per batch. If not `None`,
        overwrites `batch_size` parameter from `transform_kwargs`.
        By default, batch size is set in defaults to `64`.

        **log_file**: a stream for info messages. Default is `sys.stdout`.

        Returns a sequence of batches for training.
        """
        res = False
        config = getattr(ds, CONFIG_ATTR, None)
        if config:
            if isinstance(ds, WordCatDataset):
                for name, cfg in zip(ds.list(), config):
                    res = res \
                      and cls.transform_dataset(ds.get_dataset(name),
                                                sentences)
                    if not res:
                        break
            else:
                kwargs = config.get('transform_kwargs', {})
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
        for _ in range(1):
            if isinstance(ds, BertDataset):
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            elif isinstance(ds, WordDataset):
                kwargs = deepcopy(_DEFAULT_DATASET_TRANSFORM_KWARGS)
                if transform_kwargs:
                    loglevel = transform_kwargs.pop('loglevel', loglevel)
                    kwargs.update(transform_kwargs)
                transform_kwargs = kwargs
            else:
                break
            junky.clear_tqdm()
            res = ds.transform_collate(sentences, batch_size=batch_size,
                                       transform_kwargs=transform_kwargs,
                                       loglevel=loglevel)
        return res

    @staticmethod
    def save_dataset(ds, f, config_f=True):
        """Saves dataset to the specified file.

        Args:

        **ds**: a dataset to save.

        **f** (`str` : `file`): a file where the dataset will be saved.

        **config_f** (`str` : `file`): json config file.
        """
        if config_f is True and isinstance(f, str):
            pref, suff = os.path.splitext(f)
            config_f = pref + CONFIG_EXT
        if config_f:
            need_close = False
            if isinstance(config_f, str):
                config_f = open(config_f, 'wt', encoding='utf-8')
                need_close = True
            try:
                print(json.dumps(getattr(ds, CONFIG_ATTR, []),
                                 sort_keys=True, indent=4),
                      file=config_f)
            finally:
                if need_close:
                    config_f.close()
        ds.save(f, with_data=False)

    @classmethod
    def load_dataset(cls, f, config_f=True, emb_path=None, device=None,
                     embs=None):
        """Loads previously saved dataset with embedded tokens.

        Args:

        **f** (`str` : `file`): a file where the dataset will be loaded from.

        **config_f** (`str` : `file`): json config file.

        **emb_path**: a path where dataset to load from if you want to
        override the value from config.

        **device**: a device for the loaded dataset if you want to override
        the value from config.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If value of `emb_path` from
        **config_f** is in **embs**, the method don't load the corresponding
        model and use already loaded one.

        Returns a loaded dataset.
        """
        #ds = BaseDataset.load()
        ds = WordDataset.load(f)  # sic!

        if config_f is True:
            assert isinstance(f, str), \
                'ERROR: config_f can be True only with f of str type'
            pref, suff = os.path.splitext(f)
            config_f_ = pref + CONFIG_EXT
            if os.path.isfile(config_f_):
                config_f = config_f_
        if config_f:
            need_close = False
            if isinstance(config_f, str):
                config_f = open(config_f, 'wt', encoding='utf-8')
                need_close = True
            try:
                json.loads(config_f.read())
            finally:
                if need_close:
                    config_f.close()
        else:
            config = getattr(ds, CONFIG_ATTR, [])

        if emb_path:
            config['emb_path'] = emb_path
        if device:
            config['emb_model_device'] = device

        cls.apply_config(ds, config, embs=embs)
        return ds

    @classmethod
    def apply_config(cls, ds, config, emb_path=None, device=None, embs=None):
        """Apply config file to the dataset.

        Args:

        **ds**: a dataset to apply **config**.

        **config** (`dict` | `list([dict])`): config with model parameters.

        **emb_path**: a path where dataset to load from if you want to
        override the value from config.

        **device**: a device for the loaded dataset if you want to override
        the value from config.

        **embs**: `dict` with paths to the embeddings file as keys and
        corresponding embeddings models as values. If value of `emb_path` from
        **config** is in **embs**, the method don't load the corresponding
        model and use already loaded one.
        """
        setattr(ds, CONFIG_ATTR, config)

        if isinstance(config, dict):
            config = [config]

        assert len(config) == 1 or len(config) == len(ds.list()), \
            'ERROR: f and config_f have incompatible data'

        embs, xtrn = {} if embs is None else embs, []
        for cfg in config:
            emb_type = cfg['emb_type']
            if emb_path:
                cfg['emb_path'] = emb_path
            else:
                emb_path = cfg['emb_path']
            if device:
                cfg['emb_model_device'] = device
            else:
                device = cfg.get('emb_model_device')
            transform_kwargs = cfg.get('transform_kwargs', {})
            model = cls.load(emb_type, emb_path,
                             emb_model_device=device,
                             embs=embs)
            xtrn.append(model)

        if len(xtrn) == 1:
            xtrn = xtrn[0]
        else:
            xtrn = {x: y for x, y in zip(ds.list(), xtrn)}
        ds._push_xtrn(xtrn)

        return embs
