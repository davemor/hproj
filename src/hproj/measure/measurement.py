from hproj.data.feature_space import FeatureSpace


class Measurement:
    def __call__(
        self,
        train: FeatureSpace,
        valid: FeatureSpace,
        embeddings_train: FeatureSpace = None,
        embeddings_valid: FeatureSpace = None,
    ) -> float:
        pass


class MeasurementFactory:
    registry = {}

    @classmethod
    def register(cls, name):
        def decorator(measurement_cls):
            cls.registry[name] = measurement_cls
            measurement_cls.name = name
            return measurement_cls
        return decorator

    @classmethod
    def create(cls, name, seed, **params):
        return cls.registry[name](seed, **params)