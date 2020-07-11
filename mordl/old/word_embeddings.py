# -*- coding: utf-8 -*-
# MorDL project: Word embeddings helper
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from copy import deepcopy
from corpuscula.corpus_utils import get_root_dir
from gensim.models.fasttext import load_facebook_model
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import json
import junky
from junky.dataset import BertDataset, WordCatDataset, WordDataset
from mordl.utils import CONFIG_ATTR, LOG_FILE
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from tempfile import mkstemp
from transformers import AdamW, BertConfig, BertForTokenClassification, \
                         BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, confusion_matrix, \
                            f1_score, precision_score, recall_score

_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS = junky.kwargs(
    max_len=0, batch_size=64, hidden_ids=11,
    aggregate_hiddens_op='cat', aggregate_subtokens_op='max',
    to=junky.CPU, loglevel=2
)
_MAX_BAD_EPOCHS = 1


class WordEmbeddings:

    @staticmethod
    def bert_tune(train_sentences, train_labels, test_data=None,
                  model_name='bert-base-multilingual-cased', save_to=None,
                  device=None, max_len=512, epochs=4, batch_size=8, seed=None,
                  log_file=LOG_FILE):

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
            print("TUNE BERT MODEL '{}'. The result's model name will be '{}'"
                      .format(model_name, save_to),
                  file=log_file)

        test_sentences, test_labels = test_data if test_data else ([], [])

        if seed:
            junky.enforce_reproducibility(seed)

        def seq2ix(seq, extra_labels=None):

            seq2ix = {x: i for i, x in enumerate(sorted(set(x for x in seq
                                                              for x in x)))}

            if extra_labels:
                for tag in extra_labels:
                    seq2ix[tag] = len(seq2ix)

            return seq2ix

        cls_label, sep_label, pad_label = 'CLS', 'SEP', 'PAD'

        t2y = seq2ix(train_labels,
                     extra_labels=[cls_label, sep_label, pad_label])
        y2t = {v:k for k, v in t2y.items()}

        cls_y, sep_y, pad_y = \
            [t2y[x] for x in [cls_label, sep_label, pad_label]]

        if log_file:
            print('Loading tokenizer...', file=log_file)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        if log_file:
            print('Tokenizer is loaded. Vocab size:', tokenizer.vocab_size,
                  file=log_file)

        def prepare_corpus(sentences, labels, max_len=None):

            def tokenize_and_preserve_labels(sentence, text_labels):
                tokenized_sentence = []
                labels = []

                for word, label in zip(sentence, text_labels):

                    # Tokenize the word and count
                    # of subwords the word is broken into
                    tokenized_word = tokenizer.tokenize(word)
                    n_subwords = len(tokenized_word)

                    # Add the tokenized word to the final tokenized word list
                    tokenized_sentence.extend(tokenized_word)

                    # Add the same label to the new list
                    # of labels `n_subwords` times
                    labels.extend([t2y[label]] * n_subwords)

                return tokenized_sentence, labels

            def apply_max_len(sents, labs, max_len):
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
                                labs_.append(lab[start:end])
                    sents, labs = sents_, labs_
                return sents, labs

            return apply_max_len(*zip(*[
                tokenize_and_preserve_labels(sent, labs)
                    for sent, labs in zip(sentences, labels)
            ]), max_len)

        if log_file:
            print('Prepare corpora...', end=' ', file=log_file)
            log_file.flush()
        train_sentences_, train_labels_ = \
            prepare_corpus(train_sentences, train_labels, max_len)
        test_sentences_, test_labels_ = \
            prepare_corpus(test_sentences, test_labels, max_len)
        if log_file:
            print('done.', file=log_file)

        def collate(batch):
            sents, tags = zip(*batch)
            max_len_ = max(len(x) for x in sents)
            encoded_sents = [
                tokenizer.encode_plus(text=sent,
                                      add_special_tokens=True,
                                      max_length=max_len_ + 2,
                                      pad_to_max_length=True,
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
            output_ids = torch.tensor([
                    [cls_y] + x + [sep_y] + [pad_y] * (max_len_ - len(x))
                 for x in tags
            ])
            return input_ids, attention_masks, output_ids, lens

        class SubwordDataset(Dataset):

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
        model = BertForTokenClassification.from_pretrained(
            model_name, num_labels=len(t2y),
            output_attentions = False, output_hidden_states = False
        )
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
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]

        optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5, eps=1e-8)

        max_grad_norm = 1.

        # Total number of training steps is number of batches * number of epochs
        total_steps = len(train_loader) * epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(
        # scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        def save_finetuned_bert(model, output_dir):
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

        best_accuracy = float('-inf')
        best_test_golds, best_test_preds = [], []

        bad_epochs = 0

        junky.clear_tqdm()

        # Store the average loss after each epoch so we can plot them
        loss_values, validation_loss_values = [], []

        try:
            for epoch in range(1, epochs + 1):
                # ========================================
                #               Training
                # ========================================
                # Perform one full pass over the training set.

                # Put the model into training mode.
                model.train()
                # Reset the total loss for this epoch.
                total_loss = 0

                progress_bar = tqdm(total=len(train_loader.dataset),
                                    desc='Epoch {}'.format(epoch),
                                    file=log_file) \
                                   if log_file else \
                               None

                # Training loop
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
                    # because we have provided the `labels`.
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

                    progress_bar.set_postfix(train_loss = loss.item())
                    progress_bar.update(b_input_ids.shape[0])

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
                            # This # will return the logits rather than the
                            # loss because we have not provided labels
                            outputs = model(
                                b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels
                            )
                        # Move logits and labels to CPU
                        logits = outputs[1].detach().cpu().numpy()
                        label_ids = b_labels.to(junky.CPU).numpy()

                        # Calculate the accuracy for this batch of test
                        # sentences
                        eval_loss += outputs[0].mean().item()
                        pred_labels.extend(
                            [list(x)[1:y + 1]
                                 for x, y in zip(np.argmax(logits, axis=2),
                                                           lens)]
                        )
                        gold_labels.extend(
                            [x[1:y + 1].tolist() for x, y in zip(label_ids,
                                                                 lens)]
                        )

                        nb_eval_examples += b_input_ids.size(0)
                        nb_eval_steps += 1

                    eval_loss = eval_loss / nb_eval_steps
                    validation_loss_values.append(eval_loss)

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

                    if log_file:
                        print('Dev: accuracy = {:.8f}'.format(accuracy),
                              file=log_file)
                        print('Dev: precision = {:.8f}'.format(precision),
                              file=log_file)
                        print('Dev: recall = {:.8f}'.format(recall),
                              file=log_file)
                        print('Dev: f1_score = {:.8f}'.format(f1),
                              file=log_file)
                        print('NB: Scores may be high because of tags '
                              'stretching', file=log_file)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_test_golds, best_test_preds = \
                            gold_labels[:], pred_labels[:]

                        save_finetuned_bert(model, output_dir=save_to)
                        bad_epochs = 0

                    else:
                        bad_epochs += 1
                        if log_file:
                            print('BAD EPOCHS:', bad_epochs, file=log_file)
                        if bad_epochs >= _MAX_BAD_EPOCHS:
                            if log_file:
                                print('Maximum bad epochs exceeded. '
                                      'Process was stopped', file=log_file)
                            break

                if log_file:
                    log_file.flush()

        except RuntimeError as e:
            if e.args and e.args[0].startswith('CUDA out of memory'):
                e = RuntimeError(
                    e.args[0] + '. To avoid this, consider to decrease '
                                'batch_size or max_len value(s)',
                    *e.args[1:]
                )
            raise e

        del model

        return {'model_name': save_to,
                'best_accuracy': best_accuracy,
                'best_test_golds': best_test_golds,
                'best_test_preds': best_test_preds,
                'accuracies': accuracies,
                'precisions': precisions,
                'recalls': recalls,
                'f1s': f1s}

    @staticmethod
    def load(emb_type, emb_path, emb_model_device=None):

        if emb_type == 'bert':
            tokenizer = BertTokenizer.from_pretrained(
                emb_path, do_lower_case=False
            )
            config = BertConfig.from_pretrained(
                emb_path, output_hidden_states=True
            )
            model = BertForTokenClassification.from_pretrained(
                emb_path, config=config
            )
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
                os.remove(fn)
                model = KeyedVectors.load_word2vec_format(fn, binary=False)
            model = {x: model.vectors[y.index]
                         for x, y in model.vocab.items()}

        elif emb_type in 'ft':
            model = load_facebook_model(emb_path).wv

        elif emb_type in 'w2v':
            try:
                model = KeyedVectors.load_word2vec_format(emb_path,
                                                          binary=False)
            except UnicodeDecodeError:
                try:
                    model = KeyedVectors.load_word2vec_format(emb_path,
                                                              binary=True)
                except UnicodeDecodeError:
                    model = load_facebook_model(emb_path).wv
            model = {x: model.vectors[y.index]
                         for x, y in model.vocab.items()}

        else:
            raise ValueError('ERROR: Unknown emb_type. Allowed values: '
                             "'bert' for BERT, 'glove' for GloVe, "
                             "'ft' for fastText, 'w2v' for Word2vec")
        return model

    @classmethod
    def create_dataset(cls, sentences, emb_type='ft', emb_path=None,
                       emb_model_device=None, transform_kwargs=None,
                       next_emb_params=None):

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
                                 emb_model_device=emb_model_device)

            if emb_type == 'bert':
                model, tokenizer = emb_model
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                kwargs.update(transform_kwargs)
                junky.clear_tqdm()
                x = BertDataset(model, tokenizer)
                x.transform(sentences, **kwargs)
            else:
                x = WordDataset(
                    emb_model=emb_model,
                    vec_size=len(next(iter(emb_model.values())))
                                 if isinstance(emb_model, dict) else
                             emb_model.vector_size,
                    unk_token='<UNK>', pad_token='<PAD>'
                )
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
                'ERROR: {} class has unexpected attribute {}'
                    .format(ds.__class__, CONFIG_ATTR)
            )
        setattr(ds, CONFIG_ATTR, config)

        return ds

    @classmethod
    def transform_dataset(cls, ds, sentences, transform_kwargs=None):
        res = True
        if not transform_kwargs:
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
                    transform_kwargs = config.get('transform_kwargs', {})
        if transform_kwargs is not None:
            if isinstance(ds, BertDataset):
                kwargs = deepcopy(_DEFAULT_BERT_DATASET_TRANSFORM_KWARGS)
                kwargs.update(transform_kwargs)
                ds.transform(sentences, **kwargs)
            elif isinstance(ds, WordDataset):
                ds.transform(sentences, **transform_kwargs)
            else:
                res = False
        else:
            res = False
        return res

    @staticmethod
    def save_dataset(ds, f, config_f=True):
        if config_f is True and isinstance(f, str):
            pref, suff = os.path.splitext(f)
            config_f = pref + '.config.json'
        if config_f:
            need_close = False
            if isinstance(config_f, str):
                config_f = open(config_f, 'wt', encoding='utf-8')
                need_close = True
            try:
                print(json.dumps(getattr(ds, CONFIG_ATTR, ()),
                                 sort_keys=True, indent=4),
                      file=config_f)
            finally:
                if need_close:
                    config_f.close()
        ds.save(f, with_data=False)

    @classmethod
    def load_dataset(cls, f, config_f=True):
        ds = WordDataset.load(f)  # sic!

        if config_f is True:
            assert isinstance(f, str), \
                   'ERROR: config_f can be True only with f of str type'
            pref, suff = os.path.splitext(f)
            config_f_ = pref + '.config.json'
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
            config = getattr(ds, CONFIG_ATTR, ())

        cls.apply_config(ds, config)
        return ds

    @classmethod
    def apply_config(cls, ds, config):
        if isinstance(config, dict):
            config = [config]

        assert len(config) == 1 or len(config) == len(ds.list()), \
               'ERROR: f and config_f have incompatible data'

        embs, xtrn = {}, []
        for cfg in config:
            emb_type, emb_path = cfg['emb_type'], cfg['emb_path']
            emb_model_device = cfg.get('emb_model_device')
            transform_kwargs = cfg.get('transform_kwargs', {})
            if emb_path in embs:
                model = embs[emb_path]
            else:
                model = cls.load(emb_type, emb_path,
                                 emb_model_device=emb_model_device)
                embs[emb_path] = model
            xtrn.append(model)

        if len(xtrn) == 1:
            xtrn = xtrn[0]
        else:
            xtrn = {x: y for x, y in zip(ds.list(), xtrn)}
        ds._push_xtrn(xtrn)