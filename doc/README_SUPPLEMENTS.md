<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>

## MorDL Supplements

MorDL package contains few additional utility methods that can simplify
corpora processing.

**NB:** To use supplement methods, you should have a tagger object created.

To learn more about taggers, main pipeline and beyond, refer to other
chapters:

* [MorDL Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md)
* [POS-tagger](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md)
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md)
* [Lemmata](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md)
* [FEATS](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md)

### Table of Contents

* [Load and Save Corpora](#corp)
* [Split Corpus](#split)
* [Remove Rare Features](#rare)

### Load and Save Corpora <a name="corp"></a>

Wrappers for [***Corpuscula***](https://github.com/fostroll/corpuscula)
`Conllu.load()` and `Conllu.save()` methods:

```python
tagger.save_conllu(*args, **kwargs)
tagger.load_conllu(*args, **kwargs)
```
**args** and **kwargs** are arguments that are passed to corresponding methods
of the ***Corpuscula***'s `Conllu` class.

### Split Corpus <a name="split"></a>

Split a **corpus** in a given proportion:
```python
tagger.split_corpus(corpus, split=[.8, .1, .1], save_split_to=None,
                    seed=None, silent=False)
```
Here, **corpus** is a name of the file in
[*CoNLL-U*](https://universaldependencies.org/format.html) format or
list/iterator of sentences in
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)

**split**: `list` of sizes of the necessary **corpus** parts. If values are of
`int` type, they are interpreted as lengths of new corpora in sentences; if
values are `float`, they are proportions of a given **corpus**. The types of
the **split** values can't be mixed: they are either all `int`, or all
`float`.

**NB:** The sum of `float` values must be less or equals to 1; the sum of `int`
values can't be greater than the lentgh of the **corpus**.

**save_split_to**: `list` of file names to save the result of the **corpus**
splitting. Can be `None` (default; don't save parts to files) or its length
must be equal to the length of **split**.

**silent**: if `True`, suppress output.

Returns a list of new corpora.

### Remove Rare Features <a name="rare"></a>

If needed, you can remove rare features from train and test data. 

**Note** that this method allows you to eliminate the whole feature, not a
spesific rare label. For example, it will remove the whole `FEATS:Case`
feature, if it is unfrequent in the corpus, **not** only unfrequent ergative
'Erg' case leaving all the other cases as is.
```python
tagger.remove_rare_feats(abs_thresh=None, rel_thresh=None,
                         full_rel_thresh=None)
```
Removes feats from train and test corpora, occurence of which in the train
corpus is less then a threshold.

Args:

**abs_thresh**: remove features if their count in the train corpus is less
than this value

**rel_thresh**: remove features if their frequency with respect to total feats
count of the train corpus is less than this value

**full_rel_thresh**: remove features if their frequency with respect to the
full count of the tokens of the train corpus is less than this value
