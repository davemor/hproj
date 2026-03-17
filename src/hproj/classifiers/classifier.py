import abc

import cupy as cp

from hproj.data.feature_space import FeatureSpace


class Classifier(abc.ABC):
    # projectors take seed, **args

    @abc.abstractmethod
    def fit(self, space: FeatureSpace):
        pass

    @abc.abstractmethod
    def predict(self, space: FeatureSpace) -> cp.ndarray:
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
