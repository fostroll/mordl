<h2 align="center">MorDL: Morphological Tagger (POS, lemmata, NER etc.)</h2>

## MorDL Basics

### Initialization

Currently, MorDL has 4 different tagger types (TODO):
* POS-tagger: `UposTagger()`
* NER: `NETagger()`
* Lemmata: `LemmaTagger()`
* FEATS: `FeatsTagger()`

First of all, we need to create the tagger object. Fot example, the following
code creates a part-of-speech tagger:
```python
tagger = UposTagger()
```

### Loading Train and Test Data

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
Load development test corpus to validate on during training iterations.

Args:

**corpus** a name of file in CoNLL-U format or list/iterator of sentences in
*Parsed CoNLL-U*.

**param append** add corpus to already loaded one(s).

### Removing Rare Features

```python
tagger.remove_rare_feats(abs_thresh=None, rel_thresh=None,
                         full_rel_thresh=None)
```
Remove feats from train and test corpora, occurence of which in the train
corpus is less then a threshold.

Args:

**abs_thresh**: remove features if their count in the train corpus is less
than this value

**rel_thresh**: remove features if their frequency with respect to total feats
count of the train corpus is less than this value

**full_rel_thresh**: remove features if their frequency with respect to the
full count of the tokens of the train corpus is less than this value

### TODO