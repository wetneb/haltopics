# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

import sys
from hal import HALCorpus
import cPickle

files = sys.argv[1:]
if files:
    # Re-train the dictionary on all corpora
    dct = None
    for f in files:
        corpus = HALCorpus(f, dct=dct)
        print corpus.all_labels
    with open('data/dict-hal.pkl', 'wb') as f:
        cPickle.dump(corpus.dictionary, f)
else:
    print "Usage: hal.py corpus_1 ... corpus_n"
    print "Retrains the dictionary and saves it in data/"

