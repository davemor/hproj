from cuml.metrics.cluster import silhouette_score

from hproj.data.feature_space import FeatureSpace
from hproj.measure.measurement import Measurement, MeasurementFactory


@MeasurementFactory.register('silhouette-score')
class SilhouetteScore(Measurement):
    def __init__(self, seed):
        # self.seed = seed silhouette score is determistic so we don't need the seed
        pass

    def __call__(
        self,
        train: FeatureSpace,
        valid: FeatureSpace,
        embeddings_train: FeatureSpace = None,
        embeddings_valid: FeatureSpace = None,
    ) -> float:
        """Compute the silhouette score over the validation set.
           Note that we are using a cosine distance metric as these are image embeddings.
        """
        X = valid.features
        labels = valid.labels
        score = silhouette_score(X, labels, metric='cosine')
        return score




