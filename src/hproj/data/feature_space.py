import json
from pathlib import Path

import cupy as cp
import torch

from hproj.data.paths import EmbeddingsPath, EmbeddingsSplitPath

class FeatureSpace:
    def __init__(
        self,
        features: cp.ndarray,
        labels: cp.ndarray,
        metrics: dict | None = None,
        paths: list | None = None,
    ):
        self.features = features
        self.labels = labels
        self.metrics = {} if metrics is None else metrics
        self.paths = paths

    def measure(self, metric_fn, **kwargs):
        result = metric_fn(self.features, self.labels, **kwargs)
        self.metrics[metric_fn.__name__] = result
        return result
    
    def describe(self):
        return {'features': f'{self.features.shape}, {self.features.dtype}',
                'labels': f'{self.labels.shape}, {self.labels.dtype}',
                'metrics': self.metrics}


def load_embeddings(paths: Path | EmbeddingsSplitPath, device: str = "cpu") -> FeatureSpace:
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

    # load paths metadata if present
    paths_list = None
    paths_paths = sorted(paths.glob_paths_json())
    if paths_paths:
        paths_list = []
        for p in paths_paths:
            with open(p, "r") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                paths_list.extend(payload)
            elif isinstance(payload, dict) and "paths" in payload:
                paths_list.extend(payload["paths"])

    # convert to cupy arrays
    features = cp.asarray(features)
    labels = cp.asarray(labels)

    return FeatureSpace(features, labels, paths=paths_list)


def get_train_test(embeddings_path: EmbeddingsPath) -> tuple[FeatureSpace, FeatureSpace]:
    train = embeddings_path.split("train").load()
    test = embeddings_path.split("test").load()
    return train, test