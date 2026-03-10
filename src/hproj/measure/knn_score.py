from statistics import mean

import cupy as cp
from cuml.neighbors import KNeighborsClassifier
from cuml.metrics import accuracy_score

from hproj.data.feature_space import FeatureSpace
from hproj.measure.measurement import Measurement, MeasurementFactory


@MeasurementFactory.register('mean-knn-score')
class KNNMeanScore(Measurement):
    def __init__(self, seed, ks):
        # self.seed = seed  knn score is determistic so we don't need the seed
        self.ks = ks

    def __call__(
        self,
        train: FeatureSpace,
        valid: FeatureSpace,
        embeddings_train: FeatureSpace = None,
        embeddings_valid: FeatureSpace = None,
    ) -> float:
        """Compute the mean KNN accuracy score across multiple values of k."""

        def knn_score(k):
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(train.features, train.labels)
            pred = clf.predict(valid.features)
            return float(accuracy_score(valid.labels, pred))

        scores = [knn_score(k) for k in self.ks]
        return float(mean(scores))
