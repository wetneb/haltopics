# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

import sys
from hal import HALCorpus
from math import log
import cPickle
from maxent.cmaxent import *
from collections import defaultdict
from predict import fit_maxent, eval_model

if len(sys.argv) != 3:
    print "Usage: train.py train_corpus test_corpus"
    exit(1)

with open('data/dict-hal.pkl', 'rb') as f:
    print "Loading dictionary..."
    dct = cPickle.load(f)

train = HALCorpus(sys.argv[1], dct=dct)
test = HALCorpus(sys.argv[2], dct=dct)

print "Training..."
model = fit_maxent(train)
print "Saving..."
model.save(str('data/model.maxent'))

labels = train.all_labels
print "Labels:"
print labels

def confusion(corpus, prediction):
    mat = defaultdict(lambda: defaultdict(int))
    lineno = 0
    for (sample, label) in corpus.iter_labelled():
        if lineno % 1000 == 0:
            print lineno
        pred = prediction(sample)
        mat[label][pred] += 1
        lineno += 1
    return mat

def print_confusion(confmat, all_labels):
    mat = []
    for k1 in all_labels:
        cur_line = [confmat[k1][k2] for k2 in all_labels]
        print k1+'\t'+'\t'.join(map(str, cur_line))
        mat.append(cur_line)
    return mat

def full_f1(conf, label_counts=None):
    n = len(conf)
    true_pos = list([0])*n
    false_pos = list([0])*n
    false_neg = list([0])*n
    support = list([0])*n

    for k1 in range(n):
        true_pos[k1] = conf[k1][k1]
        false_pos[k1] = sum([conf[i][k1] for i in range(n)])-true_pos[k1]
        support[k1] = sum(conf[k1])
        false_neg[k1] = support[k1] - true_pos[k1]

    precision = list([0.])*n
    recall = list([0.])*n
    fscore = list([0.])*n
    for k in range(n):
        if true_pos[k] + false_pos[k] > 0:
            precision[k] = float(true_pos[k])/(true_pos[k] +false_pos[k])
        if true_pos[k] + false_neg[k] > 0:
            recall[k] = float(true_pos[k])/(true_pos[k] + false_neg[k])
        if precision[k] + recall[k] > 0:
            fscore[k] = 2.*(precision[k] *recall[k])/(precision[k] +recall[k])

    # Macro F1
    macro_precision = sum(precision)/n
    macro_recall = sum(recall)/n
    macro_f1 = sum(fscore)/n

    # Weighted
    nb_samples = float(sum(support))
    def weighted_sum(vec):
        return sum([vec[i]*(support[i]/nb_samples) for i in range(len(vec))])
    weighted_precision = weighted_sum(precision)
    weighted_recall = weighted_sum(recall)
    weighted_f1 = weighted_sum(fscore)

    # Random baseline F1-score
    prob = 0
    bsline = 0
    for k in range(n):
        p = float(support[k])/nb_samples
        if p > 0:
            prob -= p*log(p)
            bsline += p*p
    prob /= log(2)

    print "Entropy:"+str(prob)
    print "Random baseline on train=test:"+str(bsline)
    print "Macro precision:    "+str(macro_precision)
    print "Macro recall:       "+str(macro_recall)
    print "Macro F1:           "+str(macro_f1)
    print "Weighted precision: "+str(weighted_precision)
    print "Weighted recall:    "+str(weighted_recall)
    print "Weighted F1:        "+str(weighted_f1)

print "Evaluating..."
confmat = confusion(test, lambda x: eval_model(model, x, labels))
num_mat = print_confusion(confmat, labels)
full_f1(num_mat)
