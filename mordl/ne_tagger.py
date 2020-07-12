# -*- coding: utf-8 -*-
# MorDL project: NE tagger
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
"""
from mordl import FeatTagger


class NeTagger(FeatTagger):
    """"""

    def __init__(self):
        super().__init__('MISC:NE')
