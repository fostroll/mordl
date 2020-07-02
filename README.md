<h2 align="center">MorDL: morphological parser (POS, lemmata, NER etc.)</h2>

[![PyPI Version](https://img.shields.io/pypi/v/morra?color=blue)](https://pypi.org/project/mordl/)
[![Python Version](https://img.shields.io/pypi/pyversions/morra?color=blue)](https://www.python.org/)
[![License: BSD-3](https://img.shields.io/badge/License-BSD-brightgreen.svg)](https://opensource.org/licenses/BSD-3-Clause)

The tool provides tools to organize a pipeline for complete morphological
sentence parsing and named-entity recognition.

[TODO]
Scores on *SynTagRus*: accuracy `?%` for POS tagging; `98.74%` for lemmata
detection.

This project was making with a focus on Russian language, but it can also be
used with some other languages (European, at least).

## Installation

### pip

***MorDL*** supports *Python 3.5* or later. To install it via *pip*, run:
```sh
$ pip install mordl
```

If you currently have a previous version of ***MorDL*** installed, use:
```sh
$ pip install mordl -U
```

### From Source

Alternatively, you can also install ***MorDL*** from source of this *git
repository*:
```sh
$ git clone https://github.com/fostroll/mordl.git
$ cd mordl
$ pip install -e .
```
This gives you access to examples that are not included to the *PyPI* package.

## Usage

Input and output format for ***Morra*** is
[*CoNLL-U*](https://universaldependencies.org/format.html) when input or
output is a file, or
[*Parsed CoNLL-U*](https://github.com/fostroll/corpuscula/blob/master/doc/README_PARSED_CONLLU.md)
if it is an object. Also, it allows
[***Corpuscula***'s corpora wrappers](https://github.com/fostroll/corpuscula/blob/master/doc/README_CORPORA.md)
as input.

[MorphParser Basics](https://github.com/fostroll/morra/blob/master/doc/README_BASICS.md)

[Part of Speach Tagging](https://github.com/fostroll/morra/blob/master/doc/README_POS.md)

[Lemmata Detection](https://github.com/fostroll/morra/blob/master/doc/README_LEMMA.md)

[Morphological Feats Tagging](https://github.com/fostroll/morra/blob/master/doc/README_FEATS.md)

[Named-entity Recognition](https://github.com/fostroll/morra/blob/master/doc/README_NER.md)

[Supplements](https://github.com/fostroll/morra/blob/master/doc/README_SUPPLEMENTS.md)

## Examples

You can find them in the directory `examples` of our ***Morra*** github
repository.

## License

***Morra*** is released under the BSD License. See the
[LICENSE](https://github.com/fostroll/morra/blob/master/LICENSE) file for more
details.
