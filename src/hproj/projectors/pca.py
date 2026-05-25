from cuml import PCA

from hproj.data.feature_space import FeatureSpace
from hproj.projectors.projector import Projector, ProjectorFactory


@ProjectorFactory.register("pca")
class PCAProjector(Projector):
    def __init__(self, n_components: int, seed: int, **kwargs):
        # PCA does not use the seed because there is no stochastic element
        self.n_components = n_components
        self.kwargs = kwargs

    def fit(self, space: FeatureSpace) -> None:
        self.pca = PCA(n_components=self.n_components, **self.kwargs)
        self.pca.fit(space.features)

    def transform(self, space: FeatureSpace) -> FeatureSpace:
        features = self.pca.transform(space.features)
        return FeatureSpace(features, space.labels)
