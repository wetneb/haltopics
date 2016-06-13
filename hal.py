# -*- encoding: utf-8 -*-
from __future__ import unicode_literals

import cPickle
import codecs
import re
from gensim import corpora
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

multilingual_stop_words = set(stopwords.words('english')) | set(stopwords.words('french')) | set(['http'])

REBUILD_PROAIXY_DICT = False

rootclass_re = re.compile(r'^([A-Z\-]*)')
def get_root_class(label):
    m = rootclass_re.match(label)
    if m:
        return m.group(0)

class HALCorpus(corpora.TextCorpus):
    def __init__(self, input=None, dct=None):
        super(HALCorpus, self).__init__(input)
        self.docId = []
        self.classes = []
        self.all_labels = set()

        self.char_re = re.compile('.*[a-zA-Z]')

        if dct is None:
            self.dictionary = corpora.Dictionary(self.read())
        else:
            self.dictionary = dct
        # self.dictionary.filter_extremes()  # remove stopwords etc
 
    def __iter__(self):
        for tokens in self.read():
            yield self.dictionary.doc2bow(tokens)

    def iter_labelled(self):
        for i, tokens in enumerate(self.read()):
            if len(self.classes[i]):
                label = str(self.classes[i][0])
                # TODO more accurate label ?
                doc = self.dictionary.doc2bow(tokens)
                yield (doc, label)


    def not_a_stopword(self, w):
        return (self.char_re.match(w) is not None) and not w.startswith('//') and w not in multilingual_stop_words

    def setDocId(self, lineno, docId):
        if lineno >= len(self.docId):
            self.docId.append(docId)
        else:
            self.docId[lineno] = docId

    def setClasses(self, lineno, classes):
        if lineno >= len(self.classes):
            self.classes.append(classes)
            self.all_labels |= set(classes[:1])
        else:
            self.classes[lineno] = classes

    def read(self):
        with self.getstream() as f:
            buffer = ''
            for lineno, line in enumerate(f):
                buffer += line.decode('utf-8')[0:-1]
                if line[-1] != '\n':
                    continue
                fields = buffer.strip().split('\t')
                buffer = ""
                if len(fields) != 3:
                    print "Skipping line "+str(lineno)+" with "+str(len(fields))+" fields"
                    print "Fields:"
                    print fields
                    print "Line:"
                    print line
                    print "################"
                    continue
                topics = fields[1].split(',')
                topics = filter(lambda x: len(x) > 1, topics)
                topics = map(get_root_class, topics)
                abstract = self.tokenize_abstract(fields[2])
                self.setDocId(lineno, fields[0])
                self.setClasses(lineno, topics)
                yield abstract

    def tokenize_abstract(self, string):
        words = word_tokenize(string)
        words = map(unicode.lower, words)
        words = filter(self.not_a_stopword, words)
        return words


