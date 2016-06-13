Category prediction for HAL submissions
=======================================

This is a simple maximum entropy classifier
using bag-of-words features to predict the 
root topic of a HAL submission.

This was trained from [a dataset of abstracts from HAL
and their classifications](http://antonin.delpeuch.eu/haltopics/).

Dependencies
------------

You need to install the maxent library:
https://github.com/lzhang10/maxent

As well as other Python dependencies:
pip install -r requirements.txt

Usage
-----

To classify from a pretrained model:

   tar -zxf classifier.tgz
   python app.py # runs the app

To train a new model:

   tar -zxf corpora.tgz 
   python refresh\_dict.py corpora/dump\_hal.train corpora/dump\_hal.test
   python train.py corpora/dump\_hal.train corpora/dump\_hal.test


Licence
-------

Creative Commons Zero (CC-0).

