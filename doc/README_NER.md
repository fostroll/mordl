<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

## Named-entity Recognition

With MorDL, you can create and train biLSTM-based NER models, make predictions
and evaluate them. You can train a NER model with any custom named entities
from your training corpus. The tagger search NE tags in the feature 'NE' of
MISC field, such as 'Organization', 'Person', etc.

### Initialization

First of all, you need to create a tagger object.
```python
tagger = NeTagger(feats_clip_coef=6)
```

Args:

**feats_prune_coef** (`int`): feature prunning coefficient which allows to
eliminate all features that have a low frequency. For each UPOS tag, we
get a number of occurences of the most frequent feature from FEATS field,
divide it by **feats_prune_coef** and use features, number of occurences
of which is greater than that value, to improve the prediction quality.
* `feats_prune_coef=0` means "do not use feats";
* `feats_prune_coef=None` means "use all feats";
* default `feats_prune_coef=6`.

1. [Initialization and Data Loading](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#load)
1. [Training](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#train)
1. [Save and Load the Internal State of the Tagger](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#save)
1. [Evaluation](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#eval)
1. [Inference](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#predict)
