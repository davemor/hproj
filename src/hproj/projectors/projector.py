import abc

from hproj.data.feature_space import FeatureSpace


class Projector(abc.ABC):
    # projectors take the n_components, seed, **args

    @abc.abstractmethod
    def fit(self, space: FeatureSpace):
        pass

    @abc.abstractmethod
    def transform(self, space: FeatureSpace) -> FeatureSpace:
        pass


class ProjectorFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def decorator(projector_cls):
            cls.registry[name] = projector_cls
            projector_cls.name = name
            return projector_cls
        return decorator

    @classmethod
    def create(cls, name, n_components, seed, **params):
        return cls.registry[name](n_components, seed, **params)