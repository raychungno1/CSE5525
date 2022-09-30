# models.py

from operator import index
from sentiment_data import *
from utils import *
import math

from collections import Counter
import random
import numpy as np


class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """

    def get_indexer(self) -> Indexer:
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        """
        Extract features from a sentence represented as a list of words. Includes a flag add_to_indexer to
        :param sentence: words in the example to featurize
        :param add_to_indexer: True if we should grow the dimensionality of the featurizer if new features are encountered.
        At test time, any unseen features should be discarded, but at train time, we probably want to keep growing it.
        :return: A feature vector. We suggest using a Counter[int], which can encode a sparse feature vector (only
        a few indices have nonzero value) in essentially the same way as a map. However, you can use whatever data
        structure you prefer, since this does not interact with the framework code.
        """
        raise Exception("Don't call me, call my subclasses")

    def calculate_idf(self):
        """
        Calculates the inverse document frequency after first pass through corpus.
        """
        pass


class UnigramFeatureExtractor(FeatureExtractor):
    """
    Extracts unigram bag-of-words features from a sentence. It's up to you to decide how you want to handle counts
    and any additional preprocessing you want to do.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        if add_to_indexer:
            for word in sentence:
                self.indexer.add_and_get_index(word)
            return Counter(sentence)

        return Counter([word for word in sentence if self.indexer.contains(word)])


class BigramFeatureExtractor(FeatureExtractor):
    """
    Bigram feature extractor analogous to the unigram one.
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        double_words = [w1 + w2 for w1, w2 in zip(sentence, sentence[1:])]
        
        if add_to_indexer:
            for word in double_words:
                self.indexer.add_and_get_index(word)
            return Counter(double_words)

        return Counter([word for word in double_words if self.indexer.contains(word)])


class BetterFeatureExtractor(FeatureExtractor):
    """
    Better feature extractor...try whatever you can think of!
    """

    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.num_sentences = 0
        self.idf = Counter()

    def get_indexer(self) -> Indexer:
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        if add_to_indexer:
            self.num_sentences += 1
            self.idf.update(set(sentence))
            for word in sentence:
                self.indexer.add_and_get_index(word)
            return

        bow = Counter([word for word in sentence if self.indexer.contains(word)])
        for word in bow:
            bow[word] *= self.idf[word] / len(sentence)
        
        return bow

    def calculate_idf(self):
        for word in self.idf:
            self.idf[word] = math.log((self.num_sentences + 1) / (self.idf[word] + 1))


class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, sentence: List[str]) -> int:
        """
        :param sentence: words (List[str]) in the sentence to classify
        :return: Either 0 for negative class or 1 for positive class
        """
        raise Exception("Don't call me, call my subclasses")


class TrivialSentimentClassifier(SentimentClassifier):
    """
    Sentiment classifier that always predicts the positive class.
    """

    def predict(self, sentence: List[str]) -> int:
        return 1


class PerceptronClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        self.weights = np.zeros(len(feature_extractor.get_indexer()) + 1)
        self.feat_extractor = feature_extractor
        self.indexer = feature_extractor.get_indexer()

    def predict(self, sentence: List[str]) -> int:
        feature = self.feat_extractor.extract_features(sentence)

        weighted_sum = self.weights[-1] # Bias
        for word in feature:
            idx = self.indexer.index_of(word)
            weighted_sum += self.weights[idx] * feature[word]

        return 1 if weighted_sum > 0 else 0

    def update(self, sentence: List[str], label):
        LEARNING_RATE = 0.005
        feature = self.feat_extractor.extract_features(sentence)

        if label == 0:
            for word in feature:
                idx = self.indexer.index_of(word)
                self.weights[idx] -= LEARNING_RATE * feature[word]
            self.weights[-1] -= LEARNING_RATE
        else:
            for word in feature:
                idx = self.indexer.index_of(word)
                self.weights[idx] += LEARNING_RATE * feature[word]
            self.weights[-1] += LEARNING_RATE


class LogisticRegressionClassifier(SentimentClassifier):
    """
    Implement this class -- you should at least have init() and implement the predict method from the SentimentClassifier
    superclass. Hint: you'll probably need this class to wrap both the weight vector and featurizer -- feel free to
    modify the constructor to pass these in.
    """

    def __init__(self, feature_extractor: FeatureExtractor):
        self.weights = np.zeros(len(feature_extractor.get_indexer()) + 1)
        self.feat_extractor = feature_extractor
        self.indexer = feature_extractor.get_indexer()

    def predict(self, sentence: List[str]) -> int:
        feature = self.feat_extractor.extract_features(sentence)

        # Calculate w * x
        y = self.weights[-1] # Bias
        for word in feature:
            idx = self.indexer.index_of(word)
            y += self.weights[idx] * feature[word]

        p = 1 / (1 + math.exp(-y))

        return 1 if p > 0.5 else 0

    def update(self, sentence: List[str], label):
        feature = self.feat_extractor.extract_features(sentence)

        # Calculate w * x
        y = self.weights[-1] # Bias
        for word in feature:
            idx = self.indexer.index_of(word)
            y += self.weights[idx] * feature[word]

        p = 1 / (1 + math.exp(-y))
        
        LEARNING_RATE = 0.065 * (p if label == 0 else (1 - p))

        if label == 0:
            for word in feature:
                idx = self.indexer.index_of(word)
                self.weights[idx] -= LEARNING_RATE * feature[word]
            self.weights[-1] -= LEARNING_RATE
        else:
            for word in feature:
                idx = self.indexer.index_of(word)
                self.weights[idx] += LEARNING_RATE * feature[word]
            self.weights[-1] += LEARNING_RATE


def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    """
    Train a classifier with the perceptron.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained PerceptronClassifier model
    """

    for exs in train_exs:
        feat_extractor.extract_features(exs.words, True)
    feat_extractor.calculate_idf()

    model = PerceptronClassifier(feat_extractor)

    EPOCHS = 40
    for _ in range(EPOCHS):
        for exs in np.random.permutation(train_exs):
            label = model.predict(exs.words)

            if label != exs.label:
                model.update(exs.words, exs.label)
        
    return model


def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> LogisticRegressionClassifier:
    """
    Train a logistic regression model.
    :param train_exs: training set, List of SentimentExample objects
    :param feat_extractor: feature extractor to use
    :return: trained LogisticRegressionClassifier model
    """

    for exs in train_exs:
        feat_extractor.extract_features(exs.words, True)
    feat_extractor.calculate_idf()

    model = LogisticRegressionClassifier(feat_extractor)

    EPOCHS = 50
    for _ in range(EPOCHS):
        for exs in np.random.permutation(train_exs):
            model.update(exs.words, exs.label)
        # correct = sum(exs.label == model.predict(exs.words) for exs in train_exs)
        # print(f"Epoch {i}/{EPOCHS}")
        # print(f"Accuracy: {correct}/{len(train_exs)} ({correct / len(train_exs)}%)")

    return model


def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    """
    Main entry point for your modifications. Trains and returns one of several models depending on the args
    passed in from the main method. You may modify this function, but probably will not need to.
    :param args: args bundle from sentiment_classifier.py
    :param train_exs: training set, List of SentimentExample objects
    :param dev_exs: dev set, List of SentimentExample objects. You can use this for validation throughout the training
    process, but you should *not* directly train on this data.
    :return: trained SentimentClassifier model, of whichever type is specified
    """
    # Initialize feature extractor
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        # Add additional preprocessing code here
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        # Add additional preprocessing code here
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        # Add additional preprocessing code here
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception(
            "Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    # Train the model
    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor)
    else:
        raise Exception(
            "Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
