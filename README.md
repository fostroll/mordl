<h2 align="center">MorDL: Morphological Parser (POS, lemmata, NER etc.)</h2>

[![PyPI Version](https://img.shields.io/pypi/v/morra?color=blue)](https://pypi.org/project/mordl/)
[![Python Version](https://img.shields.io/pypi/pyversions/morra?color=blue)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD-brightgreen.svg)](https://opensource.org/licenses/BSD-3-Clause)

***MorDL*** is a tool to organize a pipeline for complete morphological
sentence parsing (POS-tagging, lemmatization, morphological feature tagging)
and named entity recognition.

[TODO]
Scores on *SynTagRus*: accuracy `?%` for POS tagging; `98.74%` for lemmata
detection.

This project was developed with a focus on Russian language, but it can also
be used with other languages (European, at least).

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

[MorDL Basics](https://github.com/fostroll/mordl/blob/master/doc/README_BASICS.md)

[POS Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_UPOS.md)

[NER](https://github.com/fostroll/mordl/blob/master/doc/README_NER.md)

[Lemma Prediction](https://github.com/fostroll/mordl/blob/master/doc/README_LEMMA.md)

[Single Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEAT.md)

[Multiple Feature Tagging](https://github.com/fostroll/mordl/blob/master/doc/README_FEATS.md)

[MorDL Supplements](https://github.com/fostroll/mordl/blob/master/doc/README_SUPPLEMENTS.md)

## Examples

You can find MorDL pipeline examples in the `examples` directory of our
***MorDL*** GitHub repository.

## License

***MorDL*** is released under the BSD License. See the
[LICENSE](https://github.com/fostroll/mordl/blob/master/LICENSE) file for more
details.
