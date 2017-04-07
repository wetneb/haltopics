#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import unicode_literals
import bottle
from predict import eval_model, doc_to_context, get_probs
import json
import cPickle
from maxent.cmaxent import MaxentModel
from hal import HALCorpus

from bottle import route, run, post, request

possible_labels = [
'INFO', 'SD', 'SDV', 'SHS', 'PHYS', 'SDE', 'QFIN', 'KGF',
'SCCO', 'STIC', 'NHF', 'SPI', 'OTHER', 'STAT', 'CHIM', 'NLIN',
'MATH', 'BOND'
    ]

readable_descriptions = {
    'CHIM':'Chimie',
    'INFO':'Informatique',
    'MATH':'Mathématiques',
    'PHYS':'Physique',
    'NLIN':'Science non linéaire',
    'SCCO':'Sciences cognitives',
    'SDE':'Sciences de l\'environnement',
    'SDU': 'Planète et Univers',
    'SHS': 'Sciences de l\'Homme et Société',
    'SDV': 'Sciences du Vivant',
    'SPI': 'Sciences de l\'ingénieur',
    'STAT': 'Statistiques',
    'QFIN': 'Économie et finance quantitative',
    'OTHER': 'Autre',
}


print "Loading model..."
model = MaxentModel()
model.load(str('data/model.maxent'))
print "Done."
print "Loading dict"
with open('data/dict-hal.pkl', 'rb') as f:
    dct = cPickle.load(f)
corpus = HALCorpus(dct=dct)
print "Done."

def lookup_category(probas):
    readable_list = []
    for lst in probas:
        if lst[0] in readable_descriptions:
            readable_list.append(
                    {'code':lst[0],
                    'proba':lst[1],
                    'desc':readable_descriptions[lst[0]]})
    return readable_list

@post('/predict')
def run_prediction():
    text = request.forms.get('text').decode('utf-8')
    tokens = corpus.tokenize_abstract(text)
    bow = corpus.dictionary.doc2bow(tokens)
    ctx = doc_to_context(bow)
    pred = eval_model(model, ctx, possible_labels)
    probs = lookup_category(get_probs(model, ctx))
    return {'decision':{'code':pred,
            'description':readable_descriptions.get(pred,'Inconnu')},
            'probabilities':probs}

@route('/')
def home():
    return '''
        <!doctype html>
        <html>
            <head>
                <title>haltopics</title>
                <script src="https://code.jquery.com/jquery-3.0.0.min.js"></script>
            </head>
            <body>
                <form action="/predict" method="post">
                    <p>Text to classify:</p>
<textarea name="text" rows="20" cols="100" id="area">
Global climate change entails many threats and challenges for the
majority of crops. Above all, a reduction in yield is expected in many
parts of the world, and drought is generally believed to represent one of
the most important negative results of climate change. Fruit crops will
certainly also suffer from the increased extension of drought conditions;
however, yield is arguably not as important for fruit as for grain crops
or oil crops. Yield does matter for fruit crops, but quality criteria are
as important if not more important. Fruits are expected to supply health
benefits and to bring hedonistic pleasures associated with specific
aromatic compounds.
</textarea><br/>
                    <input type="button" value="Reset" onclick="getElementById('area').value='';" />
                    <input value="Classify!" type="submit" />
                </form>
            </body>
        </html>
        '''

if __name__ == '__main__':
   run(host='0.0.0.0', port=6377, debug=True)

app = application = bottle.default_app()

