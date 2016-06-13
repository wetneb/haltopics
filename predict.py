# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

from maxent.cmaxent import *
from collections import defaultdict
from math import log


def doc_to_context(sample):
    return [(str(s[0]),float(s[1])) for s in sample]

def fit_maxent(corpus, lda_transform=(lambda x: x), max_samples=None):
    meModel = MaxentModel()
    meModel.begin_add_event()
    meModel.verbose = True
    idx = 0
    for sample, label in corpus.iter_labelled():
        sample = lda_transform(sample)
        context = doc_to_context(sample)
        meModel.add_event(context, label, 1)
        idx += 1
        if max_samples is not None and idx >= max_samples:
            break
    meModel.end_add_event()
    meModel.train()
    return meModel

def eval_model(model, sample, possible_labels):
    pred = model.predict([(str(x),float(c)) for x,c in sample])
    return pred

