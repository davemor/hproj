import json
from pathlib import Path

import cupy as cp
import torch

from hproj.data.folds import Fold
from hproj.data.paths import EmbeddingsPath, EmbeddingsSplitPath


class FeatureSpace:
    def __init__(
        self, features: cp.ndarray, labels: cp.ndarray, metrics: dict | None = None
    ):
        self.features = features
        self.labels = labels
        self.metrics = {} if metrics is None else metrics
        self.folds_arrays = None
        self.folds_cache = []

    def measure(self, metric_fn, **kwargs):
        result = metric_fn(self.features, self.labels, **kwargs)
        self.metrics[metric_fn.__name__] = result
        return result

    def describe(self):
        return {
            "features": f"{self.features.shape}, {self.features.dtype}",
            "labels": f"{self.labels.shape}, {self.labels.dtype}",
            "metrics": self.metrics,
        }

    def num_samples(self):
        return self.features.shape[0]

    def num_classes(self):
        return int(cp.unique(self.labels).shape[0])

    def stratified_sample_per_class(self, n_samples_per_class: int) -> "FeatureSpace":
        unique_labels = cp.unique(self.labels)
        sampled_features = []
        sampled_labels = []

        for label in unique_labels:
            label_mask = self.labels == label
            label_features = self.features[label_mask]
            label_labels = self.labels[label_mask]

            if len(label_features) > n_samples_per_class:
                indices = cp.random.choice(
                    len(label_features), n_samples_per_class, replace=False
                )
                sampled_features.append(label_features[indices])
                sampled_labels.append(label_labels[indices])
            else:
                sampled_features.append(label_features)
                sampled_labels.append(label_labels)

        return FeatureSpace(
            cp.concatenate(sampled_features), cp.concatenate(sampled_labels)
        )
    
    def stratified_sample(self, total: int) -> "FeatureSpace":
        per_class = int(total / self.num_classes())
        sampled = self.stratified_sample_per_class(per_class)
        return sampled
    

    def _make_fold(
        self, train_indexs, valid_indexs
    ) -> tuple["FeatureSpace", "FeatureSpace"]:
        train_features = self.features[train_indexs]
        train_labels = self.labels[train_indexs]
        valid_features = self.features[valid_indexs]
        valid_labels = self.labels[valid_indexs]

        train_space = FeatureSpace(train_features, train_labels)
        valid_space = FeatureSpace(valid_features, valid_labels)

        return train_space, valid_space

    def get_folds(
        self, folds: list[Fold]
    ) -> list[tuple["FeatureSpace", "FeatureSpace"]]:
        if not self.folds_cache and self.folds_arrays is None:
            self.folds = folds
            self.folds_cache = [
                self._make_fold(train_idx, valid_idx) for train_idx, valid_idx in folds
            ]
        return self.folds_cache


def load_embeddings(
    paths: Path | EmbeddingsSplitPath, device: str = "cpu"
) -> FeatureSpace:
    if isinstance(paths, Path):
        paths = EmbeddingsSplitPath(paths)

    # load the embedding shards
    embed_paths = sorted(paths.glob_embs())
    embeds = [torch.load(p).to(device) for p in embed_paths]
    features = torch.cat(embeds, dim=0)

    # load the label shards
    label_paths = sorted(paths.glob_labels())
    labels_list = [torch.load(p).to(device) for p in label_paths]
    labels = torch.cat(labels_list, dim=0)

    # TODO:we are not going to use the paths in the feature space but could add later.

    # convert to cupy arrays
    features = cp.asarray(features)
    labels = cp.asarray(labels)

    return FeatureSpace(features, labels)


def get_train_test(
    embeddings_path: EmbeddingsPath,
) -> tuple[FeatureSpace, FeatureSpace]:
    train = embeddings_path.split("train").load()
    test = embeddings_path.split("test").load()
    return train, test
