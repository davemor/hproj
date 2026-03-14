from cuml.metrics import trustworthiness

from hproj.data.feature_space import FeatureSpace
from hproj.measure.measurement import Measurement, MeasurementFactory


@MeasurementFactory.register('trustworthiness')
class Tustworthiness(Measurement):
    def __init__(self, seed):
        # self.seed = seed trustworthiness determistic so we don't need the seed
        pass

    def __call__(
        self,
        train: FeatureSpace,
        valid: FeatureSpace,
        embeddings_train: FeatureSpace = None,
        embeddings_valid: FeatureSpace = None,
    ) -> float:
        """Compute the trustworthiness over the validation set.
           Note that we are using a cosine distance metric as these are image embeddings.
        """
        score = trustworthiness(valid.features, embeddings_valid.features)  # note - cosine metric is not supported
        return score
