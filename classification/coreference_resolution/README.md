# old course work on coreference resolution

I use logitsic regression from scikit-learn on this project, but technically this is not implementation of an ML algorithm. 

This project uses the CoNLL 2012 dataset: <http://conll.cemantix.org/2012/data.html>. 

To train a model, simply run:
python corpus.py

Then to run evaluate on the output, run (NOTE: reference-coreference-scorers-8.01 is required):
bash batch_scorer.sh
