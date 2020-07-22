<h2 align="center">MorDL: Morphological Tagger (POS, lemmata, NER etc.)</h2>

## MorDL Basics

This chapter gives an overview on MorDL taggers and the basic pipeline.

### Table of Contents

1. [Initialization](#init)
2. [Load Train and Test Data](#data)
3. [Main Pipeline: Train - Predict - Evaluate](#pipeline)
4. [Save and Load Trained Models](#save)
5. [Save and Load Model's `state_dict`](#state)
6. [Save and Restore Model Backups](#backup)

### Initialization <a name="init"></a>

Currently, MorDL has 5 different tagger types. Refer to the specific tagger
documentation for detailed information:
* [POS-tagger](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md):
`UposTagger()`
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md): 
`NETagger()`
* [Lemmata](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md):
`LemmaTagger()`
* [FEAT](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md):
`FeatTagger()`
* [FEATS](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md):
`FeatsJointTagger()` and `FeatsSeparateTagger()`

First of all, you need to create the tagger object. For example, to create a
part-of-speech tagger:
```python
tagger = UposTagger()
```

### Load Train and Test Data <a name="data"></a>

After the tagger is initialized, we need to load train and test data into it.

```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
```
Loads the train corpus.

Args:

**corpus**: a name of the file in *CoNLL-U* format or list/iterator of
sentences in *Parsed CoNLL-U*.

**append** (`bool`): whether to add **corpus** to the already loaded one(s).

**test** (`float`): if not `None`, **corpus** will be shuffled and specified
part of it stored as test corpus.

**seed** (`int`): init value for the random number generator if you need
reproducibility. Used only if test is not `None`.

```python
tagger.load_test_corpus(corpus, append=False)
```
Load development test corpus for validation during training iterations.

Args:

**corpus** a name of the file in CoNLL-U format or list/iterator of sentences
in *Parsed CoNLL-U*.

**param append** add corpus to already loaded one(s).

### Main Pipeline: Train - Predict - Evaluate <a name="pipeline"></a>

Main pipeline consists of 3 steps: training - prediction - evaluation.
Parameters vary slightly for each different tagger.

To learn more about training, prediction and evaluation steps for each tagger,
refer to the corresponding chapters:

* [POS-tagger](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md)
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md)
* [Lemmata](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md)
* [FEAT](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md)
* [FEATS](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md)

### Save and Load Trained Models <a name="save"></a>

Normally, you don't need to save the model deliberately. The model is saved
during training after each successful epoch, but you can save model
configuration at any time using `.save()` method.

The saved model can be loaded back for inference with `.load()` method. 

```python
tagger.save(self, name, log_file=LOG_FILE)
```
Saves the internal state of the tagger.

Args:

**name**: a name to save with.

**log_file**: a stream for info messages. Default is `sys.stdout`.

The `.save()` method creates 4 files for the tagger: two for the model (config
and state dict) and two for the dataset (config and the internal state). All
file names start with **name** and their endings are: `.config.json` and `.pt`
for the model; `_ds.config.json` and `_ds.pt` for the dataset.

```python
tagger.load(model_class, name, device=None, dataset_device=None,
            log_file=LOG_FILE)
```
Loads tagger's internal state saved by its `.save()` method. First, you need
to initialize the model class and then load trained model parameters into it.

Args:

**model_class**: model class object.

**name** (`str`): name of the previously saved internal state.

**device**: a device for the loading model if you want to override its
previously saved value.

**dataset_device**: a device for the loading dataset if you want to
override its previously saved value.

**log_file**: a stream for info messages. Default is `sys.stdout`.

### Save and Load Model's `state_dict` <a name="state"></a>

You can save and load only model's `state_dict` using `save_state_dict` and
`load_state_dict` methods.

```python
tagger.save_state_dict(f, log_file=LOG_FILE)
```
Saves PyTorch model's state dictionary to a file to further use for model
inference.

Args:

**f** (`str` : `file`): the file where state dictionary will be saved.

**log_file**: a stream for info messages. Default is `sys.stdout`.

```python
tagger.load_state_dict(f, log_file=LOG_FILE):
```
Loads previously saved PyTorch model's state dictionary for inference.

Args:

**f**: a file from where state dictionary will be loaded.

**log_file**: a stream for info messages. Default is `sys.stdout`.


### Save and Restore Model Backups <a name="backup"></a>

Anytime, you can backup and restore internal states of trained models.

```python
o = tagger.backup()
```
Get current state.

```python
tagger.restore(o)
```
Restore current state from backup object.

### MorDL Supplements

To learn about supplement methods, supported in MorDL, refer to the
[SUPPLEMENTS](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md)
chapter.