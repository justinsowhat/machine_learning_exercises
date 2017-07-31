# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
from corpus import Document

from cStringIO import StringIO
from unittest import TestCase, main

class TrivialDocument(Document):
    """A document whose sole feature is its identity."""
    
    def features(self):
        return [id(self)]

class TrivialClassifier(Classifier):
    """A classifier that classifies based on exact feature matches."""

    def __init__(self, model={}):
        super(TrivialClassifier, self).__init__(model)

    def get_model(self): return self.seen
    def set_model(self, model): self.seen = model
    model = property(get_model, set_model)

    def train(self, instances):
        """Remember the labels associated with the features of instances."""
        for instance in instances:
            for feature in instance.features():
                self.seen[feature] = instance.label

    def classify(self, instance):
        """Classify an instance using the features seen during training."""
        for feature in instance.features():
            if feature in self.seen:
                return self.seen[feature]

class ClassifierTest(TestCase):
    def test_save_load(self):
        """Save and load a trivial model"""
        foo = TrivialDocument("foo", True)
        c1 = TrivialClassifier()
        c1.train([foo])
        model = c1.model
        stream = StringIO()
        c1.save(stream)
        self.assertGreater(len(stream.getvalue()), 0)

        c2 = TrivialClassifier()
        c2.load(StringIO(stream.getvalue()))
        self.assertEquals(c2.model, model)
        self.assertTrue(c2.classify(foo))

    def test_trivial_classifier(self):
        """Classify instances as seen or unseen"""
        corpus = [TrivialDocument(x, True) for x in ("foo", "bar", "baz")]
        classifier = TrivialClassifier()
        classifier.train(corpus)
        for x in corpus:
            self.assertTrue(classifier.classify(x))
        for y in [TrivialDocument(x) for x in ("foobar", "quux")]:
            self.assertFalse(classifier.classify(y))

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main()
