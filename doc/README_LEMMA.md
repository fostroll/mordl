<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Lemmata Prediction

In ***MorDL***, we treat lemmata prediction as a sequence labelling task,
rather than a sequence-to-sequence problem, as described in
[Straka M., 2018](https://www.aclweb.org/anthology/K18-2020.pdf), part 4.4,
and
[Kondratyuk D. and Milan Straka, 2019](https://www.aclweb.org/anthology/D19-1279.pdf),
part 2.3. Overall, lemmatization is based on edit operations from source string
to lemmatized string.

### Table of Contents

1. [Initialization and Data Loading](#init)
1. [Training](#train)
1. [Save and Load the Internal State of the Tagger](#save)
1. [Evaluation](#eval)
1. [Inference](#predict)

### Initialization and Data Loading<a name="init"></a>

First of all, you need to create the tagger object:

```python
from mordl import LemmaTagger

tagger = LemmaTagger(field='LEMMA', feats_prune_coef=6, embs=None)
```

Args:

**field** (`str`; default is `LEMMA`): the name of the *CoNLL-U* field
with values that are derivatives of the FORM field, like `'LEMMA'`.

**feats_prune_coef** (`int`; default is `6`): the feature prunning
coefficient which allows to eliminate all features that have a low
frequency. To improve the prediction quality, we get a number of
occurences of the most frequent feature from the FEATS field for each UPOS
tag, divide that number by **feats_prune_coef**, and use only those
features, the number of occurences of which is greater than that value.
* `feats_prune_coef=0` means "do not use feats";
* `feats_prune_coef=None` means "use all feats";
* default `feats_prune_coef=6`.

**embs** (`dict({str: object})`; default is `None`): the `dict` with paths
to embeddings files as keys and corresponding embedding models as values.
If the tagger needs to load any embedding model, firstly, the model is
looked up it in that `dict`.

During init, **embs** is copied to the `embs` attribute of the creating
object, and this attribute may be used further to share already loaded
embeddings with another taggers.

Afterwards, load the train and the development test corpora into the created
tagger object:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
tagger.load_test_corpus(corpus, append=False)
```
For detailed info on `.load_train_corpus()` and `.load_test_corpus()`,
refer to the
[***MorDL*** Basics: Load Train and Test Data](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#data)
chapter.

### Training<a name="train"></a>

***MorDL*** allows you to train a custom lemmata prediction model. We treat
lemmata prediction as a sequence labelling task, rather than a
sequence-to-sequence problem. The model may be either the BiLSTM or the
Transformer Encoder based. The latter may have slightly better performance
(but to achieve it, you have to tune other params too), though on very long
sentences it may cause *CUDA out of memory* error on the inference.

**NB:** By this step, you should have the tagger object created and training
data loaded.

```python
stat = tagger.train(save_as,
                    device=None, control_metric='accuracy', max_epochs=None,
                    min_epochs=0, bad_epochs=5, batch_size=TRAIN_BATCH_SIZE,
                    max_grad_norm=None, tags_to_remove=None, word_emb_type='bert',
                    word_emb_path=None, word_transform_kwargs=None,
                        # BertDataset.transform() (for BERT-descendant models)
                        # params:
                        # {'max_len': 0, 'batch_size': 64, 'hidden_ids': '10',
                        #  'aggregate_hiddens_op': 'cat',
                        #  'aggregate_subtokens_op': 'absmax', 'to': junky.CPU,
                        #  'loglevel': 1}
                        # WordDataset.transform() (for other models) params:
                        # {'check_lower': True}
                    stage1_params=None,
                        # {'lr': .0001, 'betas': (0.9, 0.999), 'eps': 1e-8,
                        #  'weight_decay': 0, 'amsgrad': False,
                        #  'max_epochs': None, 'min_epochs': None,
                        #  'bad_epochs': None, 'batch_size': None,
                        #  'max_grad_norm': None}
                    stage2_params=None,
                        # {'lr': .001, 'momentum': .9, 'weight_decay': 0,
                        #  'dampening': 0, 'nesterov': False,
                        #  'max_epochs': None, 'min_epochs': None,
                        #  'bad_epochs': None, 'batch_size': None,
                        #  'max_grad_norm': None}
                    stage3_params={'save_as': None},
                        # {'save_as': None, 'epochs': 3, 'batch_size': 8,
                        #  'lr': 2e-5, 'betas': (0.9, 0.999), 'eps': 1e-8,
                        #  'weight_decay': .01, 'amsgrad': False,
                        #  'num_warmup_steps': 3, 'max_grad_norm': 1.}
                    stages=[1, 2, 3, 1, 2], save_stages=False, load_from=None,
                    learn_on_padding=False, remove_padding_intent=False,
                    seed=None, start_time=None, keep_embs=False, log_file=LOG_FILE,
                    rnn_emb_dim=384, cnn_emb_dim=None, cnn_kernels=range(1, 7),
                    upos_emb_dim=256, emb_bn=True, emb_do=.2,
                    final_emb_dim=512, pre_bn=True, pre_do=.5,
                    lstm_layers=1, lstm_do=0, tran_layers=0, tran_heads=8,
                    post_bn=True, post_do=.4)
```
Creates and trains the LEMMA prediction model.

During training, the best model is saved after each successful epoch.

*Training's args*:

**save_as** (`str`): the name using for save the model's head. Refer
to the `.save()` method's help for the broad definition (see the
**name** arg there).

**device** (`str`, default is `None`): the device for the model. E.g.:
'cuda:0'. If `None`, we don't move the model to any device (it is
placed right where it's created).

**control_metric** (`str`; default is `accuracy`): the metric that
control training. Any that is supported by the `junky.train()` method.
In the moment, it is: 'accuracy', 'f1', 'loss', 'precision', and
'recall'.

**max_epochs** (`int`; default is `None`): the maximal number of
epochs for the model's head training (stages types `1` and `2`). If
`None` (default), the training would be linger until **bad_epochs**
has met, but no less than **min_epochs**.

**min_epochs** (`int`; default is `0`): the minimal number of training
epochs for the model's head training (stages types `1` and `2`).

**bad_epochs** (`int`; default is `5`): the maximal allowed number of
bad epochs (epochs when chosen **control_metric** is not became
better) in a row for the model's head training (stages types `1` and
`2`).

**batch_size** (`int`; default is `32`): the number of sentences per
batch for the model's head training (stages types `1` and `2`).

**max_grad_norm** (`float`; default is `None`): the gradient clipping
parameter for the model's head training (stages types `1` and `2`).

**tags_to_remove** (`dict({str: str}) | dict({str: list([str])})`;
default is `None`): the tags, tokens with those must be removed from
the corpus. It's the `dict` with field names as keys and values you
want to remove. Applied only to fields with atomic values (like
*UPOS*). This argument may be used, for example, to remove some
infrequent or just excess tags from the corpus. Note, that we remove
the tokens from the train corpus completely, not just replace those
tags to `None`.

*Word embedding params*:

**word_emb_type** (`str`; default is `'bert'`): one of (`'bert'` |
`'glove'` | `'ft'` | `'w2v'`) embedding types.

**word_emb_path** (`str`): the path to the word embeddings storage.

**word_transform_kwargs** (`dict`; default is `None`): keyword
arguments for the `.transform()` method of the dataset created for
sentences to word embeddings conversion. See the `.transform()` method
of either `junky.datasets.BertDataset` (if **word_emb_path** is
`'bert'`) or `junky.datasets.WordDataset` (otherwise) if you want to
learn allowed values for the parameter. If `None`, the `.transform()`
method use its defaults.

*Training stages params*:

**stage1_param** (`dict`; default is `None`): keyword arguments for
the `BaseModel.adjust_model_for_train()` method. If `None`, the
defaults are used. Also, you can specify here new values for the
arguments **max_epochs**, **min_epochs**, **bad_epochs**,
**batch_size**, and **max_grad_norm** that will be used only on stages
of type `1`.

**stage2_param** (`dict`; default is `None`): keyword arguments for
the `BaseModel.adjust_model_for_tune()` method. If `None`, the
defaults are used. Also, you can specify here new values for the
arguments **max_epochs**, **min_epochs**, **bad_epochs**,
**batch_size**, and **max_grad_norm** that will be used only on stages
of type `2`.

**stage3_param** (`dict`; default is `None`): keyword arguments for
the `WordEmbeddings.full_tune()` method. If `None`, the defaults are
used.

**stages** (`list([int]`; default is `[1, 2, 3, 1, 2]`): what stages
we should use during training and in which order. On the stage type
`1` the model head is trained with *Adam* optimizer; the stage type
`2` is similar, but the optimizer is *SGD*; the stage type `3` is only
relevant when **word_emb_type** is `'bert'` and we want to tune the
whole model. Stage type `0` defines the skip-stage, i.e. there would
be no real training on it. It is used when you need reproducibility
and want to continue train the model from some particular stage. In
this case, you specify the name of the model saved on that stage in
the parametere **load_from**, and put zeros into the **stages** list
on the places of already finished ones. One more time: it is used for
reproducibility only, i.e. when you put some particular value to the
**seed** param and want the data order in bathes be equivalent with
data on the stages from the past trainings.

**save_stages** (`bool`; default is `False`): if we need to keep the
best model of each stage beside of the overall best model. The names
of these models would have the suffix `_<idx>(stage<stage_type>)`
where `<idx>` is an ordinal number of the stage. We can then use it to
continue training from any particular stage number (changing next
stages or their parameters) using the parameter **load_from**. Note
that we save only stages of the head model. The embedding model as a
part of the full model usually tune only once, so we don't make its
copy.

**load_from** (`str`; default is `None`): if you want to continue
training from one of previously saved stages, you can specify the name
of the model from that stage. Note, that if your model is already
trained on stage type `3`, then you want to set param
**word_emb_path** to `None`. Otherwise, you'll load wrong embedding
model. Any other params of the model may be overwritten (and most
likely, this would cause error), but they are equivalent when the
training is just starts and when it's continues. But the
**word_emb_path** is different if you already passed stage type `3`,
so don't forget to set it to `None` in that case. (Example: you want
to repeat training on stage no `5`, so you specify in the
**load_from** param something like `'model_4(stage1)'` and set the
**word_emb_path** to `None` and the **stages_param** to
`'[0, 0, 0, 0, 2]'` (or, if you don't care of reproducibility, you
could just specify `[2]` here).

*Other options*:

**learn_on_padding** (`bool`; default is `True`): while training, we
can calculate loss either taking in account predictions made for
padding tokens or without it. The common practice is don't use padding
when calculate loss. However, we note that using padding sometimes
makes the resulting model performance slightly better.

**remove_padding_intent** (`bool`; default is `False`): if you set
**learn_on_padding** param to `False`, you may want not to use padding
intent during training at all. I.e. padding tokens would be tagged
with some of real tags, and they would just ignored during computing
loss. As a result, the model would have the output dimensionality of
the final layer less by one. On the first sight, such approach could
increase the performance, but in our experiments, such effect appeared
not always.

**seed** (`int`; default is `None`): init value for the random number
generator if you need reproducibility. Note that each stage will have
its own seed value, and the **seed** param is used to calculate these
values.

**start_time** (`float`; default is `None`): the result of
`time.time()` to start with. If `None`, the arg will be init anew.

**keep_embs** (`bool`; default is `False`): by default, after creating
`Dataset` objects, we remove word embedding models to free memory.
With `keep_embs=False` this operation is omitted, and you can use
`.embs` attribute for share embedding models with other objects.

**log_file** (`file`; default is `sys.stdout`): the stream for info
messages.

*The model hyperparameters*:

**rnn_emb_dim** (`int`; default is `384`): the internal character RNN
(LSTM) embedding dimensionality. If `None`, the layer is skipped.

**cnn_emb_dim** (`int`; default is `None`): the internal character CNN
embedding dimensionality. If `None`, the layer is skipped.

**cnn_kernels** (`list([int])`; default is `[1, 2, 3, 4, 5, 6]`): CNN
kernel sizes of the internal CNN embedding layer. Relevant if
**cnn_emb_dim** is not `None`.

**upos_emb_dim** (`int`; default is `256`): the auxiliary UPOS label
embedding dimensionality.

**emb_bn** (`bool`; default is `True`): whether batch normalization
layer should be applied after the embedding concatenation.

**emb_do** (`float`; default is `.2`): the dropout rate after the
embedding concatenation.

**final_emb_dim** (`int`; default is `512`): the output dimesionality
of the linear transformation applying to concatenated embeddings.

**pre_bn** (`bool`; default is `True`): whether batch normalization
layer should be applied before the main part of the algorithm.

**pre_do** (`float`; default is `.5`): the dropout rate before the
main part of the algorithm.

**lstm_layers** (`int`; default is `1`): the number of Bidirectional
LSTM layers. If `None`, they are not created.

**lstm_do** (`float`; default is `0`): the dropout between LSTM
layers. Only relevant, if `lstm_layers` > `1`.

**tran_layers** (`int`; default is `None`): the number of Transformer
Encoder layers. If `None`, they are not created.

**tran_heads** (`int`; default is `8`): the number of attention heads
of Transformer Encoder layers. Only relevant, if `tran_layers` > `1`.

**post_bn** (`bool`; default is `True`): whether batch normalization
layer should be applied after the main part of the algorithm.

**post_do** (`float`; default is `.4`): the dropout rate after the
main part of the algorithm.

The method returns the train statistics.

### Save and Load the Internal State of the Tagger<a name="save"></a>

To save and load the state of the tagger, use methods:
```python
tagger.save(self, name, log_file=LOG_FILE)
tagger.load(model_class, name, device=None, dataset_emb_path=None,
            dataset_device=None, log_file=LOG_FILE)
```
Normally, you don't need to call the method `.save()` because the data is
saved automatically during training. Though, there are cases when this method
could be useful. For detailed info on `.save()` and `.load()`, refer to the
[MorDL Basics: Save and Load the Internal State of the Tagger](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#save)
chapter.

### Evaluation<a name="eval"></a>

When the training is done, you may evaluate the model quality using the test
or the development test corpus:
```python
tagger.evaluate(gold, test=None, use_cdict_coef=False, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, log_file=LOG_FILE)
```

Args:

**gold**: the corpus of sentences with actual target values to score
the tagger on. May be either the name of the file in *CoNLL-U* format
or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

**test** (default is `None`): the corpus of sentences with predicted
target values. If `None` (default), the **gold** corpus will be
retagged on-the-fly, and the result will be used as the **test**.

**feats** (`str | list([str])`; default is `None`): one or several
subfields of the key-value type fields like `FEATS` or `MISC` to be
evaluated separatedly.

**label** (`str`; default is `None`): the specific label of the target
field to be evaluated separatedly, e.g. `field='UPOS', label='VERB'`
or `field='FEATS:Animacy', label='Inan'`.

**use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
we use our prediction only. If `True`, we replace our prediction to
the value returned by the `corpuscula.CorpusDict.predict_<field>()`
method if its `coef` >= `.99`. Also, you can specify your own
threshold as the value of the param.

**batch_size** (`int`; default is `64`): the number of sentences per
batch.

**split** (`int`; default is `None`): the number of lines in sentences
split. Allows to process a large dataset in pieces ("splits"). If
**split** is `None` (default), all the dataset is processed without
splits.

**clone_ds** (`bool`; default is `False`): if `True`, the dataset is
cloned and transformed. If `False`, `transform_collate` is used
without cloning the dataset. There is no big differences between the
variants. Both should produce identical results.

**log_file** (`file`; default is `sys.stdout`): the stream for info
messages.

The method prints metrics and returns evaluation accuracy.

### Inference<a name="predict"></a>

Using the trained tagger, predict lemmata for the specified corpus:
```python
tagger.predict(corpus, use_cdict_coef=False, with_orig=False,
               batch_size=BATCH_SIZE, split=None, clone_ds=False,
               save_to=None, log_file=LOG_FILE)
```

Args:

**corpus**: the corpus which will be used for the feature extraction
and predictions. May be either the name of the file in *CoNLL-U*
format or the `list`/`iterator` of sentences in *Parsed CoNLL-U*.

**use_cdict_coef** (`bool` | `float`; default is `False`): if `False`,
we use our prediction only. If `True`, we replace our prediction to
the value returned by the `corpuscula.CorpusDict.predict_<field>()`
method if its `coef` >= `.99`. Also, you can specify your own
threshold as the value of the param.

**with_orig** (`bool`; default is `False`): if `True`, instead of just
the sequence with predicted labels, return the sequence of tuples
where the first element is the sentence with predicted labels and the
second element is the original sentence. **with_orig** can be `True`
only if **save_to** is `None`.

**batch_size** (`int`; default is `64`): the number of sentences per
batch.

**split** (`int`; default is `None`): the number of lines in sentences
split. Allows to process a large dataset in pieces ("splits"). If
**split** is `None` (default), all the dataset is processed without
splits.

**clone_ds** (`bool`; default is `False`): if `True`, the dataset is
cloned and transformed. If `False`, `transform_collate` is used
without cloning the dataset. There is no big differences between the
variants. Both should produce identical results.

**save_to** (`str`; default is `None`): the file name where the
predictions will be saved.

**log_file** (`file`; default is `sys.stdout`): the stream for info
messages.

Returns the corpus with lemmata predicted.
