import abc

import cupy as cp

from hproj.data.feature_space import FeatureSpace


class Classifier(abc.ABC):
    # projectors take seed, **args

    @abc.abstractmethod
    def fit(self, train: FeatureSpace):
        pass

    def predict(self, evaluate: FeatureSpace) -> cp.ndarray:
        y_score = self.predict_proba(evaluate)
        y_pred = y_score.argmax(axis=1)
        return y_pred

    def predict_and_score(self, evaluate: FeatureSpace) -> tuple[cp.ndarray, cp.ndarray]:
        y_scores = self.predict_proba(evaluate)
        y_pred = y_scores.argmax(axis=1)
        return y_pred, y_scores

    @abc.abstractmethod
    def predict_proba(self, evaluate: FeatureSpace) -> cp.ndarray:
        pass


class ClassifierFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def decorator(classifier_cls):
            cls.registry[name] = classifier_cls
            classifier_cls.name = name
            return classifier_cls

        return decorator

    @classmethod
    def create(cls, name, seed, **params):
        return cls.registry[name](seed, **params)
