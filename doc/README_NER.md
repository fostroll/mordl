<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Named-entity Recognition

With ***MorDL***, you can create and train biLSTM-based NER models, make
predictions and evaluate them. You can train a NER model with any custom named
entities from your training corpus. The tagger searches for NE tags in the
feature 'NE' of MISC field, such as 'Organization', 'Person', etc.

### Initialization

First of all, you need to create a tagger object:
```python
from mordl import NeTagger

tagger = NeTagger(feats_clip_coef=6, embs=None)
```

Args:

**feats_prune_coef** (`int`): feature prunning coefficient which allows to
eliminate all features that have a low frequency. For each UPOS tag, we
get a number of occurences of the most frequent feature from FEATS field,
divide it by **feats_prune_coef** and use only those features, number of
occurences of which is greater than that value, to improve the prediction
quality.
* `feats_prune_coef=0` means "do not use feats";
* `feats_prune_coef=None` means "use all feats";
* default `feats_prune_coef=6`.

**embs**: `dict` with paths to the embeddings file as keys and
corresponding embeddings models as values. If tagger needs to load any
embeddings model, firstly, model is looked up it in that `dict`.

During init, **embs** is copied to the `emb` attribute of the creating
object, and this attribute may be used further to share already loaded
embeddings with another taggers.

`mordl.NeTagger` class is just a descendant of `mordl.FeatTagger` class
created with `feat='MISC:NE'` option. It's the only difference, so, we
propose to look into corresponding sections of the ancestor class:

1. [Data Loading](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#load)
1. [Training](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#train)
1. [Save and Load the Internal State of the Tagger](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#save)
1. [Evaluation](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#eval)
1. [Inference](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#predict)

**Note**, that if you labeled your training corpus with a
[*brat*](https://brat.nlplab.org/) tool, you may converts its annotations to
the format of `NeTagger` with a way provided by ***Toxine*** project:
```python
from toxine.brat import brat_to_ne

brat_to_ne(txt_fn, ann_fn, save_to=None)
```

Params **txt_fn**, **ann_fn** are paths to the *brat* `txt` and `ann` files.

Param **save_to** is a path where the result will be stored. If not specified,
the function returns the result as a generator of
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)
data.

Refer to
[***Toxine*** *brat* annotations support](https://github.com/fostroll/toxine/blob/master/doc/README_BRAT.md)
if you need more help.

Also, note, that before you can make NE tagging, you should make UPOS and
FEATS tagging. See the pipeline in our
[example notebook](https://github.com/fostroll/toxine/blob/master/examples/mordl.ipynb)

Because we assume that NE tags are found only in `'MISC:NE'` feature, a single
token may have only one Named-entity tag. If your sutuation is different, and
your tokens may have several NE tags, just create similar tagger objects of
`mordl.FeatTagger` directly. For example:
```python
tagger = FeatTagger('MISC:Address')
tagger = FeatTagger('MISC:Person')
```
Thus, you can try to tag addresses and person names even if they intersect
(e.g.: "Abraham Lincoln St").

Also, in such cases you can create only one tagger to predict all such
entities. We don't have a special class for it, but you can remove excess
features from the field containing Named-entities, or just create a new
key-value type field with only NE feats (say, call this field `'NE'`). After
that, just use
[`mordl.FeatsTagger`](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start):
```python
tagger = FeatsTagger('NE')
```
