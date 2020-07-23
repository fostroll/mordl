<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>
<a name="start"></a>

[![PyPI Version](https://img.shields.io/pypi/v/morra?color=blue)](https://pypi.org/project/mordl/)
[![Python Version](https://img.shields.io/pypi/pyversions/morra?color=blue)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD-brightgreen.svg)](https://opensource.org/licenses/BSD-3-Clause)

***MorDL*** is a tool to organize a pipeline for complete morphological
sentence parsing (POS-tagging, lemmatization, morphological feature tagging)
and named entity recognition.

[TODO]
Scores (accuracy) on *SynTagRus*: UPOS: `99.15%`; FEATS: `98.30%`;
LEMMA: `99.13%`. In all experiments we used `seed=42`. Some other `seed`
values may help to achive better results.

This project was developed with a focus on Russian language, but a few nuances
we used hardly might worsen the quality of other languages processing.

## Installation

### pip

***MorDL*** supports *Python 3.5* or later. To install via *pip*, run:
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

***MorDL's*** supports
[*CoNLL-U*](https://universaldependencies.org/format.html) (if input/output is
a file), or
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)
(if input/output is an object). Also, ***MorDL's*** allows
[***Corpuscula***'s corpora wrappers](https://github.com/fostroll/corpuscula/blob/master/doc/README_CORPORA.md)
as input.

[***MorDL*** Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md#start)

[Part of Speech Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_POS.md#start)

[Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md#start)

[Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md#start)

[Lemmata Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md#start)

[Named-entity Recognition](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md#start)

[Supplements](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md#start)

## Examples

You can find a Pyhon notebook with ***MorDL*** pipeline examples in the
`examples` directory of our ***MorDL*** GitHub repository.

## License

***MorDL*** is released under the BSD License. See the
[LICENSE](https://github.com/fostroll/mordl/blob/master/LICENSE) file for more
details.
