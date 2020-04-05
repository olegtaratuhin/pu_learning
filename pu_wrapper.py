from collections import Iterable
from math import ceil

import numpy as np
from numpy import where
from numpy.random import shuffle


class BaseLearner:

    def fit(self, x, y):
        pass

    def predict(self, x: Iterable) -> Iterable:
        pass

    def predict_proba(self, x: Iterable) -> Iterable:
        pass


class PuWrapper(BaseLearner):
    """
    Wraps probabilistic binary classifier to learn with method proposed by Elkan et al.
    """

    def __init__(self, estimator: BaseLearner, holdout_ratio: float, positive_label=1, negative_label=0):
        """
        Create PU learning wrapper around estimator
        :param estimator: any estimator with sklean-like api
        :param holdout_ratio: portion of dataset to use as unlabeled
        :param positive_label: mapping for positive class label
        :param negative_label: mapping for negative class label
        """
        self.estimator = estimator
        self.holdout_ratio = holdout_ratio
        self.positive_label = positive_label
        self.negative_label = negative_label
        self._c = None

    def fit(self, x: np.array, y: np.array) -> None:
        """
        Fit estimator using PU approach
        :param x: iterable with features
        :param y: iterable with labels (1 for positive)
        :return: Nothing
        """
        positive = where(y == self.positive_label)[0]
        shuffle(positive)

        holdout_size = ceil(len(positive) * self.holdout_ratio)
        holdout = positive[:holdout_size]

        x_holdout = x[holdout]
        keep = list(set(np.arange(len(y))) - set(holdout))
        try:
            x_holdout = x_holdout[:, keep]
        except IndexError:
            pass

        try:
            x_kernel = x[:, keep]
            x_kernel = x_kernel[keep]
        except IndexError:
            x_kernel = x[keep]

        y_kernel = np.delete(y, holdout)

        self.estimator.fit(x_kernel, y_kernel)

        holdout_prediction = self.estimator.predict_proba(x_holdout)
        try:
            holdout_prediction = holdout_prediction[:, 1]
        except IndexError:
            pass

        c = np.mean(holdout_prediction)
        self._c = c

    def predict_proba(self, x: np.array) -> np.array:
        """
        Get probabilistic estimates of labels
        :param x: iterable with features
        :return: probabilistic predictions
        """
        predictions = self.estimator.predict_proba(x)
        try:
            predictions = predictions[:, 1]
        except IndexError:
            pass

        return predictions / self._c

    def predict(self, x: np.array, threshold: float = 0.5, positive=1, negative=0) -> np.array:
        """
        Get predicted labels
        :param x: iterable with features
        :param threshold: threshold for classification
        :param positive: positive label mapping
        :param negative: negative label mapping
        :return: predicted labels
        """
        return np.array([self.positive_label if p >= threshold else self.negative_label for p in self.predict_proba(x)])
