from cuml.manifold import UMAP

from hproj.data.feature_space import FeatureSpace
from hproj.projectors.projector import Projector, ProjectorFactory


@ProjectorFactory.register("umap")
class UMAPProjector(Projector):
    def __init__(self, n_components, seed, **kwargs):
        self.n_components = n_components
        self.seed = seed
        self.kwargs = kwargs
        self.SUBSAMPLE_RATIO = 0.1  # NOTE - this is not a hyperparameter, UMAP can fit on 10% of the data

    def fit(self, space: FeatureSpace) -> None:
        self.umap = UMAP(
            n_components=self.n_components, random_state=self.seed, **self.kwargs
        )
        sampled = space.stratified_sample(space.num_samples * self.SUBSAMPLE_RATIO)
        self.umap.fit(sampled)

    def transform(self, space: FeatureSpace) -> FeatureSpace:
        features = self.umap.transform(space.features)
        return FeatureSpace(features, space.labels)
