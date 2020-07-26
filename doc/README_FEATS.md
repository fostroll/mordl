<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Multiple Feature Tagging

MorDL supports single and multiple feature taggers. In this chapter, we cover
a multiple feature tagger `mordl.FeatsTagger` that allows to predict all
content of key-value type fields in a single pass.

For a single feature tagger, refer to
[Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
chapter.

### Table of Contents

1. [Initialization and Data Loading](#init)
1. [Training](#train)
1. [Save and Load the Internal State of the Tagger](#save)
1. [Evaluation](#eval)
1. [Inference](#predict)
1. [Separate Feats Tagger](#separ)

`mordl.FeatsTagger` implies that target classes are compiled of all different
feature tag combinations that are present in the training set. These classes
are predicted jointly.

### Initialization and Data Loading<a name="init"></a>

First of all, you need to create a tagger object:
```python
from mordl import FeatsTagger

tagger = FeatsTagger(field='FEATS')
```

Args:

**field** (`str`): a name of the *CoNLL-U* key-value type field, content
of which needs to be predicted. With the tagger, you can predict only
key-value type fields, like FEATS.

Afterwards, load train and development test corpora into the created tagger
object:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
tagger.load_test_corpus(corpus, append=False)
```
For detailed info on `.load_train_corpus()` and `.load_test_corpus()`,
refer to
[***MorDL*** Basics: Load Train and Test Data](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#data)
chapter.

### Training<a name="train"></a>

***MorDL*** allows you to train a custom BiLSTM joint multiple feature
prediction model.

**NB:** By this step, you should have a tagger object created and training
data loaded.

```python
stat = tagger.train(save_as,
                    device=None, epochs=None, min_epochs=0, bad_epochs=5,
                    batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
                    max_grad_norm=None, tags_to_remove=None,
                    word_emb_type='bert', word_emb_model_device=None,
                    word_emb_path=None, word_emb_tune_params=None,
                    word_transform_kwargs=None, word_next_emb_params=None,
                    rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
                    upos_emb_dim=200, emb_out_dim=512, lstm_hidden_dim=256,
                    lstm_layers=3, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
                    bn3=True, do3=.4, seed=None, log_file=LOG_FILE)
```
Creates and trains a key-value type field tagger model.

During training, the best model is saved after each successful epoch.

*Training's args*:

**save_as** (`str`): the name used for save. Refer to the `.save()`
method's help for the broad definition (see the **name** arg there).

**device**: device for the model. E.g.: 'cuda:0'.

**epochs** (`int`): number of epochs to train. If `None` (default),
train until `bad_epochs` is met, but no less than `min_epochs`.

**min_epochs** (`int`): minimum number of training epochs. Default is
`0`

**bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
during which the selected **control_metric** does not improve) in a
row. Default is `5`.

**batch_size** (`int`): number of sentences per batch. For training,
default `batch_size=32`.

**control_metric** (`str`): metric to control training. Any that is
supported by the `junky.train()` method. In the moment it is:
'accuracy', 'f1' and 'loss'. Default `control_metric=accuracy`.

**max_grad_norm** (`float`): gradient clipping parameter, used with
`torch.nn.utils.clip_grad_norm_()`.

**tags_to_remove** (`dict({str: str})|dict({str: list([str])})`):
tags, tokens with those must be removed from the corpus. It's a `dict`
with field names as keys and with value you want to remove. Applied
only to fields with atomic values (like UPOS). This argument may be
used, for example, to remove some infrequent or just excess tags from
the corpus. Note, that we remove the tokens from the train corpus as a
whole, not just replace those tags to `None`.

*Word embedding params*:

**word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v')
embedding types.

**word_emb_model_device**: the torch device where the model of word
embeddings are placed. Relevant only with embedding types, models of
which use devices (currently, only 'bert'). `None` means
**word_emb_model_device** = **device**

**word_emb_path** (`str`): path to word embeddings storage.

**word_emb_tune_params**: parameters for word embeddings finetuning.
For now, only BERT embeddings finetuning is supported with
`mordl.WordEmbeddings.bert_tune()`. So, **word_emb_tune_params** is a
`dict` of keyword args for this method. You can replace any except
`test_data`.

**word_transform_kwargs** (`dict`): keyword arguments for
`.transform()` method of the dataset created for sentences to word
embeddings conversion. See the `.transform()` method of
`junky.datasets.BertDataset` for the the description of the
parameters.

**word_next_emb_params**: if you want to use several different
embedding models at once, pass the parameters of the additional model
as a dictionary with keys
`(emb_path, emb_model_device, transform_kwargs)`; or a list of such
dictionaries if you need more than one additional model.

*Model hyperparameters*:

**rnn_emb_dim** (`int`): character RNN (LSTM) embedding
dimensionality. If `None`, the layer is skipped.

**cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
`None`, the layer is skipped.

**cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
`cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
**cnn_emb_dim**.

**upos_emb_dim** (`int`): auxiliary embedding dimensionality for UPOS
labels. Default `upos_emb_dim=200`.

**emb_out_dim** (`int`): output embedding dimensionality. Default
`emb_out_dim=512`.

**lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
`lstm_hidden_dim=256`.

**lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
`lstm_layers=3`.

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

*Other options*:

**seed** (`int`): init value for the random number generator if you
need reproducibility.

**log_file**: a stream for info messages. Default is `sys.stdout`.

The method returns the train statistics.

### Save and Load the Internal State of the Tagger<a name="save"></a>

To save and load the state of the tagger, use methods:
```python
tagger.save(self, name, log_file=LOG_FILE)
tagger.load(model_class, name, device=None, dataset_device=None,
            log_file=LOG_FILE)
```
Normally, you don't need to call the method `.save()` because the data is
saved automatically during training. Though, there are cases when this method
could be useful. For detailed info on `.save()` and `.load()`, refer to
[MorDL Basics: Save and Load the Internal State of the Tagger](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#save)
chapter.

### Evaluation<a name="eval"></a>

When the training is done, you may evaluate the model quality using the test
or the development test corpora:
```python
tagger.evaluate(gold, test=None, feats=None, label=None,
                batch_size=BATCH_SIZE, split=None, clone_ds=False,
                log_file=LOG_FILE)
```

Args:

**gold**: a corpus of sentences with actual target values to score the
tagger on. May be either a name of the file in *CoNLL-U* format or a
list/iterator of sentences in *Parsed CoNLL-U*.

**test**: a corpus of sentences with predicted target values. If
`None`, the **gold** corpus will be retagged on-the-fly, and the
result will be used as **test**.

**feats** (`str|list([str])`): one or several feature names of the
key-value type fields like `FEATS` or `MISC` to be evaluated.

**label** (`str`): specific label of the target feature value to be
evaluated, e.g. `label='Inan'`. If you specify a value here, you must
also specify the feature name as **feats** param (e.g.:
`feats='Animacy'`). Note, that in that case the param **feats** must
contain only one feature name.

**batch_size** (`int`): number of sentences per batch. Default
`batch_size=64`.

**split** (`int`): number of lines in each split. Allows to process a
large dataset in pieces ("splits"). Default `split=None`, i.e. process
full dataset without splits.

**clone_ds** (`bool`): if `True`, the dataset is cloned and
transformed. If `False`, `transform_collate` is used without cloning
the dataset. There is no big difference between the variants. Both
should produce identical results.

**log_file**: a stream for info messages. Default is `sys.stdout`.

The method prints metrics and returns evaluation accuracy.

### Inference<a name="predict"></a>

Using the trained tagger, for the specified corpus, predict content of the
FEATS field:
```python
tagger.predict(self, corpus, with_orig=False, batch_size=BATCH_SIZE,
               split=None, clone_ds=False, save_to=None, log_file=LOG_FILE):
```

Args:

**corpus**: a corpus which will be used for feature extraction and
predictions. May be either a name of the file in *CoNLL-U* format or
list/iterator of sentences in *Parsed CoNLL-U*.

**with_orig** (`bool`): if `True`, instead of only a sequence with
predicted labels, returns a sequence of tuples where the first element
is a sentence with predicted labels and the second element is the
original sentence. `with_orig` can be `True` only if `save_to` is
`None`. Default `with_orig=False`.

**batch_size** (`int`): number of sentences per batch. Default
`batch_size=64`.

**split** (`int`): number of lines in each split. Allows to process a
large dataset in pieces ("splits"). Default `split=None`, i.e. process
full dataset without splits.

**clone_ds** (`bool`): if `True`, the dataset is cloned and
transformed. If `False`, `transform_collate` is used without cloning
the dataset. There is no big differences between the variants. Both
should produce identical results.

**save_to**: file name where the predictions will be saved.

**log_file**: a stream for info messages. Default is `sys.stdout`.

Returns corpus with feature keys and values predicted in the FEATS
field.

## Separate Feats Tagger<a name="separate"></a>

Aside from the joint feats tagger described above, **MorDL** provides a
tagger that predicts all features of key-value type fields separately:
```python
from mordl.feats_tagger import FeatsSeparateTagger

tagger = FeatsSeparateTagger(field='FEATS')
```

The tagger creates separate models for each feat and uses them all serially to
fill the content of the field predicting. Obviously, it needs much more memory
to load (if you use BERT embeddings, you will have separate BERT model for
each feat, too) and much more time to train. By an order more. However, the
evaluation results of it are slightly lower than the results of the joint
tagger. Maybe, just a couple of separate feats would have a better accuracy.
Thereby, we can't find many reasons to use this class. However, we kept this
class in the project.

The separate tagger has the same methods as the joint tagger. But the
arguments of the base methods slightly differ. You can explore it with Python
`help()` command:
```python
help(tagger.train)
help(tagger.save)
help(tagger.load)
help(tagger.evaluate)
help(tagger.predict)
```
or even:
```python
help(tagger)
```

Also, note that you can't change the device names of models during loading.
If you need to do so (really, you need, because all models together can hardly
fit on a single GPU), you have to edit configuration files of the separate
models.
