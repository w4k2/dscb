from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import BaseEnsemble
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from math import pi, exp
import numpy as np


class DSCB(ClassifierMixin, BaseEnsemble):

    def __init__(self,
                 base_estimator=KNeighborsClassifier(),
                 number_of_chunks=10,
                 balance_ratio=0.45,
                 oversampling=SMOTE(),
                 undersampling=RandomUnderSampler(),
                 hard_voting=False,
                 is_weighted=True,
                 xi=0.01,
                 min_param=5,
                 delay=5):

        self.base_estimator = base_estimator
        self.number_of_chunks = number_of_chunks
        self.undersampling = undersampling
        self.oversampling = oversampling
        self.balance_ratio = balance_ratio

        self.is_weighted = is_weighted
        self.xi = xi
        self.min_param = min_param
        self.delay = delay

        self.hard_voting = hard_voting

        self.ensemble_ = []
        self.stored_X = []
        self.stored_y = []
        self.number_of_features = None

        self.minority_name = None
        self.majority_name = None
        self.classes_ = None
        self.label_encoder = None

        self.iterator = 0

    def partial_fit(self, X, y, classes=None):

        self.ensemble_ = []

        # ________________________________________
        # Initial preperation

        if classes is None and self.classes_ is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(y)
            self.classes_ = self.label_encoder.classes
        elif self.classes_ is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(classes)
            self.classes_ = classes

        if classes[0] is "positive":
            self.minority_name = self.label_encoder.transform(classes[0])
            self.majority_name = self.label_encoder.transform(classes[1])
        elif classes[1] is "positive":
            self.minority_name = self.label_encoder.transform(classes[1])
            self.majority_name = self.label_encoder.transform(classes[0])

        y = self.label_encoder.transform(y)

        if self.minority_name is None or self.majority_name is None:
            self.minority_name, self.majority_name = self.minority_majority_name(y)
            self.number_of_features = len(X[0])

        # ________________________________________
        # Get stored data

        new_X, new_y = [], []

        for tmp_X, tmp_y in zip(self.stored_X, self.stored_y):
            new_X.extend(tmp_X)
            new_y.extend(tmp_y)

        new_X.extend(X)
        new_y.extend(y)

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        # ________________________________________
        # Undersample and store new data

        und_X, und_y = self.undersampling.fit_resample(X, y)

        self.stored_X.append(und_X)
        self.stored_y.append(und_y)

        if len(self.stored_X) > self.number_of_chunks:
            del self.stored_X[0]
            del self.stored_y[0]

        # ________________________________________
        # Oversample when below ratio

        minority, majority = self.minority_majority_split(new_X, new_y, self.minority_name, self.majority_name)
        ratio = len(minority)/len(majority)

        if ratio < self.balance_ratio:
            new_X, new_y = self.oversampling.fit_resample(new_X, new_y)

        # ________________________________________
        # Train classifiers

        self.bagging(X, y, xi=self.xi)

    def bagging(self, X, y, xi):

        weights = []
        new_X, new_y = [], []
        iter = len(self.stored_X)

        for tmp_X, tmp_y in zip(self.stored_X, self.stored_y):

            w_maj = (2*pi/exp((iter*xi)/2))
            if self.iterator < self.delay:
                w_min = (2*pi/exp(((iter-self.min_param)*xi)/2))
            else:
                w_min = (2*pi/exp((iter*xi)/2))

            wr = np.where(tmp_y == 0, w_maj, w_min)
            weights.extend(wr)

            new_X.extend(tmp_X)
            new_y.extend(tmp_y)
            iter -= 1

        wr = np.where(y == 0, w_maj, w_min)
        weights.extend(wr)

        weights = np.array(weights)
        weights = weights / weights.sum()

        new_X.extend(X)
        new_y.extend(y)

        new_X = np.array(new_X)
        new_y = np.array(new_y)

        indicies = np.array(list(range(new_X.shape[0])))
        for i in range(10):
            if self.is_weighted:
                bag = np.random.choice(indicies, new_X.shape[0], replace=True, p=weights)
            else:
                bag = np.random.choice(indicies, new_X.shape[0], replace=True)

            self.ensemble_.append(clone(self.base_estimator).fit(new_X[bag], new_y[bag]))

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : array-like, shape (n_samples, )
            The predicted classes.
        """

        if self.hard_voting:
            preds_ = np.asarray([clf.predict(X) for clf in self.ensemble_]).T
            prediction = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=preds_)
        else:
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)

        return self.classes_[prediction]

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority : array-like, shape = [n_samples, n_features]
            Minority class samples.
        majority : array-like, shape = [n_samples, n_features]
            Majority class samples.
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns the name of minority and majority class

        Parameters
        ----------
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        minority_name : object
            Name of minority class.
        majority_name : object
            Name of majority class.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name
