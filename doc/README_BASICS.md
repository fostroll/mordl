<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## MorDL Basics

This chapter provides an overview on ***MorDL*** taggers and the basic pipeline.

### Table of Contents

1. [Initialization](#init)
1. [Load Train and Test Data](#data)
1. [Usage: Training - Evaluation - Inference](#usage)
1. [Save and Load the Internal State of the Tagger](#save)
1. [***MorDL*** Supplements](#suppl)

### Initialization<a name="init"></a>

Currently, ***MorDL*** has a bunch of tools for morphological tagging. Refer
to the specific tagger documentation for detailed information:
* [`mordl.UposTagger`: Part of Speech Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_POS.md#start)
* [`mordl.FeatTagger`: Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
* [`mordl.FeatsTagger`: Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)
(along with `morld.feats_tagger.FeatsSeparateTagger`)
* [`mordl.LemmaTagger`: Lemmata Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)
* [`mordl.NeTagger`: Named-entity Recognition](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

First of all, you need to create the tagger object. For example, to create the
part-of-speech tagger, call:

```python
from mordl import UposTagger

tagger = UposTagger()
```

The exhausting list of the constructor paramters, also as the descriptions of
the taggers of other types, one can find in the corresponding chapters.

### Load Train and Test Data<a name="data"></a>

After the tagger is initialized, we need to load the train and test corpora
into it.

To load the train corpus, use:
```python
tagger.load_train_corpus(corpus, append=False, test=None, seed=None)
```

Args:

**corpus**: either the name of the file in *CoNLL-U* format or the
`list`/`iterator` of sentences in *Parsed CoNLL-U*.

**append** (`bool`; default is `False`): whether to add the **corpus**
to the already loaded one(s).

**test** (`float`; default is `None`): if not `None`, the **corpus**
will be shuffled and the specified part of it stored as a test corpus.

**seed** (`int`; default is `None`): the init value for the random
number generator if you need reproducibility. Only used if test is not
`None`.

To load the development test corpus, call:
```python
tagger.load_test_corpus(corpus, append=False)
```
This corpus is used for validation during training iterations. It is not
mandatory to load it, but without validation it's hard to stop training, when
the quality of the model is at its highest.

Args:

**corpus**: either the name of the file in *CoNLL-U* format or the
`list`/`iterator` of sentences in *Parsed CoNLL-U*.

**append** (`bool`; default is `False`): whether to add the **corpus**
to the already loaded one(s).

### Usage: Training - Evaluation - Inference<a name="usage"></a>

All our taggers contain 3 main methods: `.train()`, `.evaluate()` and
`.predict()`. Parameters vary slightly for each tagger and are described in
the corresponding chapters:
* [POS-tagger](https://github.com/fostroll/mordl/blob/master/doc/README_POS.md#start)
* [FEAT](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
* [FEATS](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)
* [Lemmata](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

### Saveing and Loading of the Internal State of the Tagger<a name="save"></a>

Normally, you don't need to save the state of the tagger intentionally. During
training, the state is saved automatically after each successful epoch.
However, you can save the state configuration at any time, if you need. For
example, you could want to save, if you load the model or/and the dataset to
the device(s) that differ from the device(s) used during training, and you
wish to save it back with renewed parameters. You can do so with:
```python
tagger.save(name, log_file=LOG_FILE)
```

Args:

**name**: the name to save with.

**log_file** (`file`; default is `sys.stdout`): the stream for info
messages.

The method creates the directory **name** that contains 5 files: two
for the tagger's model (`model.config.json` and `model.pt`) and two
for its dataset (`ds.config.json` and `ds.pt`). The 5th file
(`cdict.pickle`) is an internal state of the
[`corpuscula.CorpusDict`](https://github.com/fostroll/corpuscula/blob/master/doc/README_CDICT.md)
object that is used by the tagger as a helper.

`*.config.json` files contain parameters for creation of the objects.
They are editable, but you are allowed to change only the device name.
Any other changes most likely won't allow the tagger to load.

The saved model can be loaded back for inference with:
```python
tagger.load(name, device=None, dataset_emb_path=None, dataset_device=None,
            log_file=LOG_FILE)
```

Args:

**name** (`str`): the name of the previously saved internal state.

**device** (`str`; default is `None`): the device for the loaded model
if you want to override the value from the config.

**dataset_emb_path** (`str`; default is `None`): the path where the
dataset's embeddings to load from if you want to override the value
from the config.

**dataset_device** (`str`; default is `None`): the device for the
loaded dataset if you want to override the value from the config.

**log_file** (`file`; default is `sys.stdout`): the stream for info
messages.

### MorDL Supplements<a name="suppl"></a>

Aside from the basic methods, ***MorDL*** contain several supplement utilities
that sometimes may be helpful. You may find their description in the
[MorDL Supplements](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md)
chapter.
