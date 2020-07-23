<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## MorDL Basics

This chapter gives an overview on ***MorDL*** taggers and the basic pipeline.

### Table of Contents

1. [Initialization](#init)
1. [Load Train and Test Data](#data)
1. [Usage: Train - Predict - Evaluate](#pipeline)
1. [Save and Load the Internal State of the Tagger](#save)

### Initialization <a name="init"></a>

Currently, ***MorDL*** has a bunch of tools for morphological tagging. Refer
to the specific tagger documentation for detailed information:
* [`mordl.UposTagger`: Part of Speech Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_POS.md#start)
* [`mordl.FeatTagger`: Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
* [`mordl.FeatsTagger`: Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)
(along with `morld.feats_tagger.FeatsSeparateTagger`)
* [`mordl.LemmaTagger`: Lemmata Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)
* [`mordl.NeTagger`: Named-entity Recognition](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

First of all, you need to create the tagger object. For example, to create a
part-of-speech tagger, call:
```python
tagger = UposTagger()
```

For taggers of other types, you can find a description of their creation in
corresponding chapters.

### Load Train and Test Data <a name="data"></a>

After the tagger is initialized, we need to load train and test corpora into
it.

To load the train corpus, use:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
```

Args:

**corpus**: a name of the file in *CoNLL-U* format or list/iterator of
sentences in *Parsed CoNLL-U*.

**append** (`bool`): whether to add **corpus** to the already loaded one(s).

**test** (`float`): if not `None`, **corpus** will be shuffled and specified
part of it stored as test corpus.

**seed** (`int`): init value for the random number generator if you need
reproducibility. Used only if test is not `None`.

To load the development test corpus, call:
```python
tagger.load_test_corpus(corpus, append=False)
```
This corpus is used for validation during training iterations. It is not
mandatory to load it, but without validation it's hard to stop training in
time, when the quality of model is the highest.

Args:

**corpus** a name of the file in *CoNLL-U* format or list/iterator o
sentences in *Parsed CoNLL-U*.

**param append** add corpus to already loaded one(s).

### Usage: Train - Predict - Evaluate <a name="usage"></a>

All our taggers contain 3 main methods: `.train()`, `.predict()` and
`.evaluate()`. Parameters vary slightly for each tagger and described in the
corresponding chapters:
* [POS-tagger](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md#start)
* [FEAT](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
* [FEATS](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)
* [Lemmata](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

### Save and Load the Internal State of the Tagger <a name="save"></a>

Normally, you don't need to save the state of the tagger intentionally. During
training, the state is saved automatically after each successful epoch.
However, you can save the state configuration at any time, if you need. For
example, you could want to do it, if you load model or/and dataset to
device(s) that is differ from the device(s) used during training, and wish to
save it back with renewed parameters. You can do it with:
```python
tagger.save(name, log_file=LOG_FILE)
```

Args:

**name**: a name to save with.

**log_file**: a stream for info messages. Default is `sys.stdout`.

The method creates a directory **name** that contains 5 files: two for
tagger's model (`model.config.json` and `model.pt`) and two for its
dataset (`ds.config.json` and `ds.pt`). The 5th file (`cdict.pickle`)
is an internal state of
[`corpuscula.CorpusDict`](https://github.com/fostroll/corpuscula/blob/master/doc/README_CDICT.md)
object that is used by the tagger as a helper.

`*.config.json` files contain parameters for objects' creation. They
are editable, but you allowed to change only the device name. Any
other changes most likely won't allow the tagger to load.

The saved model can be loaded back for inference with:
```python
tagger.load(name, device=None, dataset_device=None, log_file=LOG_FILE)
```

Args:

**name** (`str`): name of the previously saved internal state.

**device**: a device for the loading model if you want to override its
previously saved value.

**dataset_device**: a device for the loading dataset if you want to
override its previously saved value.

**log_file**: a stream for info messages. Default is `sys.stdout`.

### MorDL Supplements

Aside from basic methods, ***MorDL*** contain several supplement utilities
that sometimes may be helpful. You may find their description in the
[MorDL Supplements](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md)
chapter.
