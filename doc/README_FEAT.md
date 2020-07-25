<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Single Feature Tagging

***MorDL*** supports single and multiple feature taggers. In this chapter, we
cover a single feature tagger `mordl.FeatTagger`. This class is used to
predict only one certain feature of field with dict-type data. It can be
useful for many purposes. For example, our Named-entity tagger is just
descendant of this class, that predicts NE feature located in the MISC field.

For multiple feature tagger, refer to
[Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)
chapter.

### Table of Contents

1. [Initialization and Data Loading](#init)
1. [Training](#train)
1. [Save and Load the Internal State of the Tagger](#save)
1. [Evaluation](#eval)
1. [Inference](#predict)

### Initialization and Data Loading <a name="init"></a>

First of all, you need to create a tagger object:
```python
tagger = FeatTagger(feat, feats_prune_coef=6)
```

Args:

**feat** (`str`): the name of the feat which needs to be predicted by the
tagger. May contains prefix, separated by a colon (`:`). In that case, the
prefix treat as a field name. Otherwise, we get `'FEATS'` as a field name.
Examples: `'Animacy'`; `'MISC:NE'`.

**feats_prune_coef** (`int`): feature prunning coefficient which allows to
eliminate all features that have a low frequency. For each UPOS tag, we
get a number of occurences of the most frequent feature from FEATS field,
divide it by **feats_prune_coef** and use features, number of occurences
of which is greater than that value, to improve the prediction quality.
* `feats_prune_coef=0` means "do not use feats";
* `feats_prune_coef=None` means "use all feats";
* default `feats_prune_coef=6`.

**NB**: the argument is relevant only if **feat** is not from FEATS field.

Afterwards, load train and development test corpora into the tagger object
created:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
tagger.load_test_corpus(corpus, append=False)
```
For detailed info on `.load_train_corpus()` and `.load_test_corpus()`,
refer to
[***MorDL*** Basics: Load Train and Test Data](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#data)
chapter.

### Training <a name="train"></a>

***MorDL*** allows you to train a custom BiLSTM single morphological feature
prediction model.

**NB:** By this step you should have a tagger object created and training data
loaded.

```python
tagger = train(save_as,
               device=None, epochs=None, min_epochs=0, bad_epochs=5,
               batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
               max_grad_norm=None, tags_to_remove=None,
               word_emb_type='bert', word_emb_model_device=None,
               word_emb_path=None, word_emb_tune_params=None,
               word_transform_kwargs=None, word_next_emb_params=None,
               rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
               upos_emb_dim=300, emb_out_dim=512, lstm_hidden_dim=256,
               lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
               bn3=True, do3=.4, seed=None, log_file=LOG_FILE):
```
Creates and trains a feature tagger model.

During training, the best model is saved after each successful epoch.

*Training's args*:

**save_as** (`str`): the name using for save. Refer to the `.save()`
method's help for the broad definition (see the **name** arg there).

**device**: device for the model. E.g.: 'cuda:0'.

**epochs** (`int`): number of epochs to train. If `None` (default),
train until `bad_epochs` is met, but no less than `min_epochs`.

**min_epochs** (`int`): minimum number of training epochs. Default is
`0`

**bad_epochs** (`int`): maximum allowed number of bad epochs (epochs
when selected **control_metric** is became not better) in a row.
Default is `5`.

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
embedding models at once, pass parameters of the additional model as a
dictionary with keys `(emb_path, emb_model_device, transform_kwargs)`;
or a list of such dictionaries if you need more than one additional
model.

*Model hyperparameters*:

**rnn_emb_dim** (`int`): character RNN (LSTM) embedding
dimensionality. If `None`, the layer is skipped.

**cnn_emb_dim** (`int`): character CNN embedding dimensionality. If
`None`, the layer is skipped.

**cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
`cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None`
**cnn_emb_dim**.

**upos_emb_dim** (`int`): auxiliary embedding dimensionality for UPOS
labels. Default `upos_emb_dim=300`.

**emb_out_dim** (`int`): output embedding dimensionality. Default
`emb_out_dim=512`.

**lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
`lstm_hidden_dim=256`.

**lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
`lstm_layers=2`.

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

### Save and Load the Internal State of the Tagger <a name="save"></a>

To save and load state of the tagger, use methods:
```python
tagger.save(self, name, log_file=LOG_FILE)
tagger.load(model_class, name, device=None, dataset_device=None,
            log_file=LOG_FILE)
```
Normally, you don't need to call the method `.save()` because the data is
saved automatically during training. Though, there are cases when this method
is useful. For detailed info on `.save()` and `.load()`, refer to
[MorDL Basics: Save and Load the Internal State of the Tagger](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#save)
chapter.

### Evaluation <a name="eval"></a>

When the training has done, you may evaluate its quality using the test or
development test corpora:
```python
tagger.evaluate(gold, test=None, label=None, batch_size=BATCH_SIZE,
                 split=None, clone_ds=False, log_file=LOG_FILE)
```
Args:

**gold**: a corpus of sentences with actual target values to score the
tagger on. May be either a name of the file in *CoNLL-U* format or
list/iterator of sentences in *Parsed CoNLL-U*.

**test**: a corpus of sentences with predicted target values. If
`None`, the **gold** corpus will be retagged on-the-fly, and the
result will be used **test**.

**label** (`str`): specific label of the target feat to be evaluated,
e.g. `label='Inan'` for tagger created with `field='Animacy'` option.

**batch_size** (`int`): number of sentences per batch. Default
`batch_size=64`.

**split** (`int`): number of lines in each split. Allows to process a
large dataset in pieces ("splits"). Default `split=None`, i.e. process
full dataset without splits.

**clone_ds** (`bool`): if `True`, the dataset is cloned and
transformed. If `False`, `transform_collate` is used without cloning
the dataset. There is no big differences between the variants. Both
should produce identical results.

**log_file**: a stream for info messages. Default is `sys.stdout`.

The method prints metrics and returns evaluation accuracy.

### Inference <a name="predict"></a>

Using the trained tagger, for the specified corpus, predict values of certain
feature of the FEATS field:
```python
tagger.predict(corpus, with_orig=False, batch_size=BATCH_SIZE,
               split=None, clone_ds=False, save_to=None, log_file=LOG_FILE)
```

Args:

**corpus**: a corpus which will be used for feature extraction and
predictions. May be either a name of the file in *CoNLL-U* format or
list/iterator of sentences in *Parsed CoNLL-U*.

**with_orig** (`bool`): if `True`, instead of only a sequence with
predicted labels, returns a sequence of tuples where the first element
is a sentence with predicted labels and the second element is original
sentence. `with_orig` can be `True` only if `save_to` is `None`.
Default `with_orig=False`.

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

Returns corpus with predicted values of certain feature in the FEATS
field.
