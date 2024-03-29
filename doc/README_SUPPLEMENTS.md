<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## ***MorDL*** Supplements

The `mordl` package contains a few additional utility methods that can be
helpful. To use supplement methods, in most cases you don't have to create a
tagger object. Howewer, you have to import a tagger class from the `mordl`
package. Let it be `mordl.UposTagger`, although equally well it can be a class
of any other tagger:
```python
from mordl import UposTagger
```

To learn more about the usage of the taggers, refer to other chapters:

* [***MorDL*** Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#start)
* [POS Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md#start)
* [NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)
* [Lemma Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)
* [Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)
* [Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)

### Table of Contents

1. [Load and Save *CoNLL-U* Files](#conllu)
1. [Load Word Embeddings](#embs)
1. [Split Corpus](#split)
1. [Remove Rare Features](#rare)
1. [String Comparison](#diff)
1. [Evaluation Script of *CoNLL 2018*](#conll18)

### Load and Save *CoNLL-U* Files<a name="conllu"></a>

Usually, we use methods of the
[***Corpuscula***](https://github.com/fostroll/corpuscula) project to work
with [*CoNLL-U*](https://universaldependencies.org/format.html) format.
However, for convenience, we include wrappers for ***Corpuscula***'s
`Conllu.load()` and `Conllu.save()` methods to our project:

```python
UposTagger.save_conllu(*args, **kwargs)
UposTagger.load_conllu(*args, **kwargs)
```
**args** and **kwargs** are arguments that are passed to corresponding methods
of the
[`corpuscula.Conllu`](https://github.com/fostroll/corpuscula/blob/master/doc/README_CONLLU.md)
class.

### Load Word Embeddings<a name="embs" />

The package has the method for loading pretrained embeddings model of any
supported type:
```python
emb_model = UposTagger.load_word_embeddings(emb_type, emb_path,
                                            emb_model_device=None, embs=None)
```

Args:

**emb_type**: (`str`) one of the supported embeddings types. Allowed
values: `'bert'` for *BERT*, `'ft'` for *FastText*, `'glove'` for *Glove*,
`'w2v'` for *Word2vec*.

**emb_path** (`str`): path to the embeddings file.

**emb_model_device** (`str`; default is `None`): relevant with
`emb_type='bert'`. The device where the *BERT* model will be loaded.

**embs** (`dict({str: object})`; default is `None`): the `dict` with paths
to embeddings files as keys and corresponding embedding models as values.
If the tagger needs to load any embedding model, firstly, the model is
looked up it in that `dict`.

Returns the embeddings model.

### Split Corpus<a name="split"></a>

Split a **corpus** in a given proportion:
```python
UposTagger.split_corpus(corpus, split=[.8, .1, .1], save_split_to=None,
                        seed=None, silent=False)
```
Here, **corpus** is a name of the file in *CoNLL-U* format or a
`list`/`iterator` of sentences in
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)

**split**: `list` of sizes of the necessary **corpus** parts. If values are of
`int` type, they are interpreted as lengths of new corpora in sentences; if
values are of `float` type, then they are proportions of a given **corpus**.
The types of the **split** values can't be mixed: they are either all `int` or
all `float`.

**NB:** The sum of `float` values must be less or equal to `1`; the sum of
`int` values can't be greater than the lentgh of the **corpus**.

**save_split_to**: `list` of file names to save the result of the **corpus**
splitting. Can be `None` (default; don't save parts to files) or its length
must be equal to the length of the **split** `list`.

**silent**: if `True`, suppress output.

Returns a list of new corpora.

### Remove Rare Features<a name="rare"></a>

This method is used when you already have the tagger created (let it be the
`tagger` variable), and if before starting the training you want to remove
rare FEATS keys from the already loaded the train and the corpora for to
denoise your training data.

**Note** that this method allows you to eliminate the whole feature, not a
specific rare label. For example, it will remove the whole `FEATS:Case`
feature, if it is infrequent in the corpus, but you **can not** remove just
one infrequent ergative `'Erg'` case, leaving all the other cases as is.

```python
tagger.remove_rare_feats(abs_thresh=None, rel_thresh=None,
                         full_rel_thresh=None)
```
Removes feats from the train and test corpora, occurence of which in the train
corpus is less then a threshold.

Args:

**abs_thresh**: remove features if their count in the train corpus is less
than this value.

**rel_thresh**: remove features if their frequency with respect to total
number of feats in the train corpus is less than this value.

**full_rel_thresh**: remove features if their frequency with respect to the
full number of the tokens in the train corpus is less than this value.

### String Comparison<a name="diff"></a>

Methods of this section are contained in the `mordl.LemmaTagger` class. So,
we need to import it first:
```python
from mordl import LemmaTagger
```

To detect common parts and the distinctions between two strings, call:
```python
form_pref, form_root, form_suff, lemma_pref, lemma_root, lemma_suff = \
    LemmaTagger.find_affixes(form, lemma, lower=False)
```
Finds the longest common part of the given **form** and **lemma**
and returns concurrent and distinct parts of both **form** and
**lemma** given.

Args:

**lower** (`bool`; default is `False`): if `True`, return values will
be always in lower case. Elsewise, we compare strings in lower case
but return values will be in original case.

Returns the prefix, the common part, the suffix/flexion of the
**form**, as well as the prefix, the common part, the suffix/flexion
of the **lemma** (i.e. the `tuple` of 6 `str` values).

The wrapper for the method
[`get_editops`](https://rawgit.com/ztane/python-Levenshtein/master/docs/Levenshtein.html#Levenshtein-editops)
of the
[`python-Levenshtein`](https://pypi.org/project/python-Levenshtein/)
package. We convert the output of the method to the format that is useful for
our purposes.
```python
ops = LemmaTagger.get_editops(str_from, str_to,
                              allow_replace=True, allow_copy=True)
```
Gets edit operations from **str_from** to **str_to** according to
Levenstein distance. Supported edit operations are: `'delete'`,
`'insert'`, `'replace'`, `'copy'`.

Args:

**str_from** (`str`): the source string.

**str_to** (`str`): the target string.

**allow_replace** (`bool`; default is `True`): whether to allow the
**replace** edit operation.

**allow_copy** (`bool`; default is `True`): whether to allow the
**copy** edit operation.

Returns the `tuple` of edit operations that is needed to transform
**str_from** to **str_to**.

To apply these operations to the string, call:
```python
str_to = LemmaTagger.apply_editops(str_from, ops)
```

Args:

**str_from** (`str`): the source string to apply edit operations.

**ops** (`tuple([str])`): the `tuple`/`list` with edit operations.

Returns **str_from** with **ops** applied.

### <a name="conll18"></a>Official Evaluation Script of *CoNLL 2018 Shared Task*

Because the official
[*CoNLL18 UD Shared Task*](https://universaldependencies.org/conll18/results.html)
evaluation script is used often to evaluate morphological and syntactic
parsers, we included it in our project (as `conll18_ud_eval.py` in
`/mordl/mordl/lib` directory of our GitHub repository. To simplify its usage,
we also made a wrapper for it:

```python
from mordl import conll18_ud_eval

conll18_ud_eval(gold_file, system_file, verbose=True, counts=False)
```

Args:

**gold_file**: Name of the CoNLL-U file with the gold data.

**system_file**: Name of the CoNLL-U file with the predicted.

**verbose** (`bool`): Print all metrics.

**counts** (`bool`): Print raw counts of correct/gold/system/aligned words
instead of prec/rec/F1 for all metrics.

If `verbose=False`, only the official CoNLL18 UD Shared Task evaluation
metrics are printed.

If `verbose=True` (default), more metrics are printed (as precision,
recall, F1 score, and in case the metric is computed on aligned words
also accuracy on these):
- Tokens: how well do the gold tokens match system tokens.
- Sentences: how well do the gold sentences match system sentences.
- Words: how well can the gold words be aligned to system words.
- UPOS: using aligned words, how well does UPOS match.
- XPOS: using aligned words, how well does XPOS match.
- UFeats: using aligned words, how well does universal FEATS match.
- AllTags: using aligned words, how well does UPOS+XPOS+FEATS match.
- Lemmas: using aligned words, how well does LEMMA match.
- UAS: using aligned words, how well does HEAD match.
- LAS: using aligned words, how well does HEAD+DEPREL(ignoring subtypes)
match.
- CLAS: using aligned words with content DEPREL, how well does
HEAD+DEPREL(ignoring subtypes) match.
- MLAS: using aligned words with content DEPREL, how well does
HEAD+DEPREL(ignoring subtypes)+UPOS+UFEATS+FunctionalChildren(DEPREL+UPOS+UFEATS)
match.
- BLEX: using aligned words with content DEPREL, how well does
HEAD+DEPREL(ignoring subtypes)+LEMMAS match.

If `count=True`, raw counts of correct/gold_total/system_total/aligned
words are printed instead of precision/recall/F1/AlignedAccuracy for all
metrics.
