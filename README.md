<h2 align="center">MorDL: Morphological Tagger (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

[![PyPI Version](https://img.shields.io/pypi/v/mordl?color=blue)](https://pypi.org/project/mordl/)
[![Python Version](https://img.shields.io/pypi/pyversions/mordl?color=blue)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD-brightgreen.svg)](https://opensource.org/licenses/BSD-3-Clause)

***MorDL*** is a tool to organize the pipeline for complete morphological
sentence parsing (POS-tagging, lemmatization, morphological feature tagging)
and Named-entity recognition.

Scores (accuracy) on *SynTagRus* test dataset: UPOS: `99.35%`; FEATS: `98.87%`
(tokens), `99.31%` (tags); LEMMA: `99.50%`. In all experiments, we used
`seed=42`. Some other `seed` values may help to achive better results. Models'
hyperparameters are also allowed to tune.

The validation with the
[official evaluation script](http://universaldependencies.org/conll18/conll18_ud_eval.py)
of
[CoNLL 2018 Shared Task](https://universaldependencies.org/conll18/results.html):
* For the inference on the *SynTagRus* test corpus, when predicted fields were
emptied and all other fields were stayed intact, the scores are the same as
outlined above.
* The inference of UPOS - FEATS - LEMMA taggers applied serially resulted with
scores: UPOS: `99.35%`; UFeats: `98.36%`; AllTags: `98.21`; Lemmas: `98.88%`.

For completeness, we included that script in our distribution, so you can use
it for your model evaluation, too. To simplify it, we also made a wrapper 
[`mordl.conll18_ud_eval`](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md#conll18)
for it.

## Installation

### pip

***MorDL*** supports *Python 3.6* and *Transformers 4.3.3* or later. To
install via *pip*, run:
```sh
$ pip install mordl
```

If you currently have a previous version of ***MorDL*** installed, run:
```sh
$ pip install mordl -U
```

### From Source

Alternatively, you can install ***MorDL*** from the source of this *git
repository*:
```sh
$ git clone https://github.com/fostroll/mordl.git
$ cd mordl
$ pip install -e .
```
This gives you access to examples that are not included in the *PyPI* package.

## Usage

Our taggers use separate models, so they can be used independently. But to
achieve best results FEATS tagger uses UPOS tags during training. And LEMMA
and NER taggers use both UPOS and FEATS tags. Thus, for a fully untagged
corpus, the tagging pipeline is serially applying the taggers, like shown
below (assuming that our goal is NER and we already have trained taggers of
all types):

```python
from mordl import UposTagger, FeatsTagger, NeTagger

tagger_u, tagger_f, tagger_n = UposTagger(), FeatsTagger(), NeTagger()
tagger_u.load('upos_model')
tagger_f.load('feats_model')
tagger_n.load('misc-ne_model')

tagger_n.predict(
    tagger_f.predict(
        tagger_u.predict('untagged.conllu')
    ), save_to='result.conllu'
)
```

Any tagger in our pipeline may be replaced with a better one if you have it.
The weakness of separate taggers is that they take more space. If all models
were created with BERT embeddings, and you load them in memory simultaneously,
they may eat up to 9Gb on GPU. If it does not fit to your GPU, during loading,
you can use params **device** and **dataset_device** to distribute your models
on various GPUs. Alternatively, if you need just to tag some corpus once, you
may load models serially:

```python
tagger = UposTagger()
tagger.load('upos_model')
tagger.predict('untagged.conllu', save_to='result_upos.conllu')
del tagger  # just for sure
tagger = FeatsTagger()
tagger.load('feats_model')
tagger.predict('result_upos.conllu', save_to='result_feats.conllu')
del tagger
tagger = NeTagger()
tagger_n.load('misc-ne_model')
tagger.predict('result_feats.conllu', save_to='result.conllu')
del tagger
```

Don't use identical names for input and output file names when you call the
`.predict()` methods. Normally, there will be no problem, because the methods
by default load all the input file in memory before tagging. But if the input
file is large, you may want to use the **split** parameter for the methods
handle the file by parts. In that case, saving of the first part of the
tagging data occurs before loading next. So, identical names will entail data
loss.

The training process is also simple. If you have training corpora and you
don't want any experiments, just run:

```python
from mordl import UposTagger

tagger = UposTagger()
tagger.load_train_corpus(train_corpus)
tagger.load_test_corpus(dev_corpus)

stat = tagger.train('upos_model', device='cuda:0',
                    stage3_params={'save_as': 'upos_bert_model'})
```

It is a training pipeline for the UPOS tagger; pipelines for other taggers are
identical.

For a more complete understanding of ***MorDL*** toolkit usage, refer to the
Python notebook with the pipeline example in the `examples` directory of the
***MorDL*** GitHub repository. Also, the detailed descriptions are available
in the docs:

[***MorDL*** Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#start)

[Part of Speech Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_POS.md#start)

[Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)

[Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)

[Lemmata Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)

[Named-entity Recognition](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

[Supplements](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md#start)

Also, you can find training pipelines for different taggers in our
[example notebook](https://github.com/fostroll/mordl/blob/master/examples/mordl.ipynb).

This project was developed with the focus on Russian language, but a few
nuances we use for it are unlikely to worsen the quality of processing other
languages.

***MorDL's*** supports
[*CoNLL-U*](https://universaldependencies.org/format.html) (if input/output is
a file), or
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)
(if input/output is an object). Also, ***MorDL's*** allows
[***Corpuscula***'s corpora wrappers](https://github.com/fostroll/corpuscula/blob/master/doc/README_CORPORA.md)
as input.

## License

***MorDL*** is released under the BSD License. See the
[LICENSE](https://github.com/fostroll/mordl/blob/master/LICENSE) file for more
details.
