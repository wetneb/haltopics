Category prediction for HAL submissions
=======================================

**This repository has been archived**. We recommend to use [Annif]( https://github.com/NatLibFi/Annif) instead.

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
<pre>
# extract the pre-trained classifier
tar -zxf classifier.tgz
# run the app
python app.py
</pre>

To train a new model:
<pre>
# extract the corpora
tar -zxf corpora.tgz 
# create the dictionary covering both train and test corpora (maps words to numbers)
python refresh\_dict.py corpora/dump\_hal.train corpora/dump\_hal.test
# train the model on the first corpus, test on the second
python train.py corpora/dump\_hal.train corpora/dump\_hal.test
</pre>


Licence
-------

Creative Commons Zero (CC-0).

