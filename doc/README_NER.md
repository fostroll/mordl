<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Named-entity Recognition

With MorDL, you can create and train biLSTM-based NER models, make predictions
and evaluate them. You can train a NER model with any custom named entities
from your training corpus.

### Table of Contents

1. [Initialization and Data Loading](#init)
1. [Train](#train)
1. [Save and Load the Internal State of the Tagger](#save)
1. [Predict](#predict)
1. [Evaluate](#eval)

### Initialization and Data Loading <a name="init"></a>

First of all, you need to create a tagger object.
```python
tagger = NETagger(field='MISC:NE', cdict=None, feats_clip_coef=6,
				  log_file=LOG_FILE)
```
Creates an `NETagger` object.

Args:

**field** (`str`): the name of the field which needs to be predicted by the
training tagger. May contain up to 3 elements, separated by a colon (`:`).
Format is: `'<field name>:<feat name>:<replacement for None>'`. The
replacement is used during training as a filler for a fields without a value
so that we could predict them, too. In the *CoNLL-U* format the replacer is a
`'_'` sign, so we use it as a default replacement. Normally, you wouldn't need
to change this parameter. Examples:<br/>
`'UPOS'` - predict the *UPOS* field;<br/>
`'FEATS:Animacy'` - predict only the *Animacy* feat of the *FEATS* field;<br/>
`'FEATS:Animacy:_O'` - likewise the above, but if feat value is `None`, it
will be replaced by `'_O'` during training;<br/>
`'XPOS::_O'` - predict the *XPOS* field and use `'_O'` as replacement for
`None`.

**cdict**: [TODO]

**feats_clip_coef** (`int`): feature clipping coefficient which allows to
eliminate all features that have a lower frequency than
`<most frequent feature frequency>` divided by `feats_clip_coef`.
* `feats_clip_coef=0` means "do not use feats"
* `feats_clip_coef=None` means "use all feats"
Relevant only if `cdict` is specified and `field` is not from `FEATS`.

**log_file**: a stream for info messages. Default is `sys.stdout`.

Afterwards, load train and test data into the tagger object:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
tagger.load_test_corpus(corpus, append=False)
```
For detailed info on `.load_train_corpus()` and `.load_test_corpus()`,
refer to
[MorDL Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md)
chapter.

### Train <a name="train"></a>

***MorDL*** allows you to train a custom NER model.

**NB:** By this step you should have a tagger object created and training data
loaded.

```python
tagger.train(save_as, device=None, epochs=None, min_epochs=0, bad_epochs=5,
			 batch_size=TRAIN_BATCH_SIZE, control_metric='accuracy',
			 max_grad_norm=None, tags_to_remove=None,
			 word_emb_type='bert', word_emb_model_device=None,
			 word_emb_path=None, word_emb_tune_params=None,
			 word_transform_kwargs=None, word_next_emb_params=None,
			 rnn_emb_dim=None, cnn_emb_dim=None, cnn_kernels=range(1, 7),
			 upos_emb_dim=None, emb_out_dim=512, lstm_hidden_dim=256,
			 lstm_layers=2, lstm_do=0, bn1=True, do1=.2, bn2=True, do2=.5,
			 bn3=True, do3=.4, seed=None, log_file=LOG_FILE)
```
Creates and trains a named entity tagger model.

During training, the best model is saved after each successful epoch.

Args:

**save_as** (`str`): the name of the tagger used for save. As a result, 4
files will be created after training: two for tagger's model (config and state
dict) and two for the dataset (config and the internal state). All file names
use **save_as** as a prefix and their endings are: `.config.json` and `.pt`
for the model; `_ds.config.json` and `_ds.pt` for the dataset.

**device**: device for the model. E.g.: 'cuda:0'.

**epochs** (`int`): number of epochs to train. If `None`, train until
`bad_epochs` is met, but no less than `min_epochs`.

**min_epochs** (`int`): minimum number of training epochs.

**bad_epochs** (`int`): maximum allowed number of bad epochs (epochs when
selected **control_metric** is became not better) in a row. Default
`bad_epochs=5`.

**batch_size** (`int`): number of sentences per batch. For training, default
`batch_size=32`.

**control_metric** (`str`): metric to control training. Any that is supported
by the `junky.train()` method. Currently, options are: 'accuracy', 'f1' and
'loss'. Default `control_metric=accuracy`.

**max_grad_norm** (`float`): gradient clipping parameter, used with
`torch.nn.utils.clip_grad_norm_()`.

**tags_to_remove** (`list([str])|dict({str: list([str])})`): tags, tokens with
those must be removed from the corpus. May be a `list` of tag names or a
`dict` of `{<feat name>: [<feat value>, ...]}`. This argument may be used, for
example, to remove some infrequent tags from the corpus. Note, that we remove
the tokens from the train corpus as a whole, not just replace those tags to
`None`.

**word_emb_type** (`str`): one of ('bert'|'glove'|'ft'|'w2v') embedding types.

**word_emb_path** (`str`): path to word embeddings storage.

**word_emb_model_device**: the torch device where the model of word embeddings
is placed. Relevant only with embedding types, models of which use devices
(currently, only 'bert').

**word_emb_tune_params**: parameters for word embeddings finetuning. For now,
only BERT embeddings finetuning is supported with
`mordl.WordEmbeddings.bert_tune()`. So, **word_emb_tune_params** is a `dict`
of keyword args for this method. You can replace any except `test_data`.

**word_transform_kwargs** (`dict`): keyword arguments for `.transform()`
method of the dataset created for sentences to word embeddings conversion. See
the `.transform()` method of `junky.datasets.BertDataset` for the the
description of the parameters.

**word_next_emb_params**: if you want to use several different embedding
models at once, pass parameters of the additional model as a dictionary with
keys `(emb_path, emb_model_device, transform_kwargs)`; or a list of such
dictionaries if you need more than one additional model.

**rnn_emb_dim** (`int`): character RNN (LSTM) embedding dimensionality. If
`None`, the layer is skipped.

**cnn_emb_dim** (`int`): character CNN embedding dimensionality. If `None`,
the layer is skipped.

**cnn_kernels** (`list([int])`): CNN kernel sizes. By default,
`cnn_kernels=[1, 2, 3, 4, 5, 6]`. Relevant with not `None` **cnn_emb_dim**.

**upos_emb_dim** (`int`): auxiliary UPOS label embedding dimensionality.
Default `upos_emb_dim=60`.

**emb_out_dim** (`int`): output embedding dimensionality. Default
`emb_out_dim=512`.

**lstm_hidden_dim** (`int`): Bidirectional LSTM hidden size. Default
`lstm_hidden_dim=256`.

**lstm_layers** (`int`): number of Bidirectional LSTM layers. Default
`lstm_layers=1`.

**lstm_do** (`float`): dropout between LSTM layers. Only relevant, if
`lstm_layers` > `1`.

**bn1** (`bool`): whether batch normalization layer should be applied after
the embedding layer. Default `bn1=True`.

**do1** (`float`): dropout rate after the first batch normalization layer
`bn1`. Default `do1=.2`.

**bn2** (`bool`): whether batch normalization layer should be applied after
the linear layer before LSTM layer. Default `bn2=True`.

**do2** (`float`): dropout rate after the second batch normalization layer
`bn2`. Default `do2=.5`.

**bn3** (`bool`): whether batch normalization layer should be applied after
the LSTM layer. Default `bn3=True`.

**do3** (`float`): dropout rate after the third batch normalization layer
`bn3`. Default `do3=.4`.

**seed** (`int`): init value for the random number generator if you need
reproducibility.

**log_file**: a stream for info messages. Default is `sys.stdout`.

Returns the train statistics.

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

### Predict <a name="predict"></a>

Using the trained corpus, predict tags for the specified corpus:
```python
tagger.predict(corpus, with_orig=False, batch_size=BATCH_SIZE, split=None,
			   clone_ds=False, save_to=None, log_file=LOG_FILE)
```
Predicts tags in the `MISC:NE` fields of the corpus.

Args:

**corpus**: input corpus which will be used for feature extraction and
predictions.

**with_orig** (`bool`): if `True`, instead of only a sequence with predicted
labels, returns a sequence of tuples where the first element is a sentence
with predicted labels and the second element is original sentence labels.
`with_orig` can be `True` only if `save_to` is `None`. Default
`with_orig=False`.

**batch_size** (`int`): number of sentences per batch. Default
`batch_size=64`.

**split** (`int`): number of lines in each split. Allows to split a large
dataset into several parts. Default `split=None`, i.e. process full dataset
without splits.

**clone_ds** (`bool`): if `True`, the dataset is cloned and transformed. If
`False`, `transform_collate` is used without cloning the dataset.

**save_to**: directory where the predictions will be saved.

**log_file**: a stream for info messages. Default is `sys.stdout`.

Returns corpus with tag predictions in the `MISC:NE` field.

### Evaluate <a name="eval"></a>

When predictions are ready, evaluate predicitons on the development test set
based on the gold corpus:
```python
tagger.evaluate(gold, test=None, label=None, batch_size=BATCH_SIZE,
                split=None, clone_ds=False, log_file=LOG_FILE)
```
Evaluates predicitons on the development test set.

Args:

**gold** (`tuple(<sentences> <labels>)`): corpus with actual target tags.

**test** (`tuple(<sentences> <labels>)`): corpus with predicted target tags.
If `None`, predictions will be created on-the-fly based on the `gold` corpus.

**label** (`str`): specific label of the target field to be evaluated, e.g.
`label='GEO'`.

**batch_size** (`int`): number of sentences per batch. Default
`batch_size=64`.

**split** (`int`): number of lines in each split. Allows to split a large
dataset into several parts. Default `split=None`, i.e. process full dataset
without splits.

**clone_ds** (`bool`): if `True`, the dataset is cloned and transformed. If
`False`, `transform_collate` is used without cloning the dataset.

**log_file**: a stream for info messages. Default is `sys.stdout`.

Prints metrics and returns evaluation accuracy.
