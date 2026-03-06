import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pathlib import Path


load_dotenv()


@dataclass
class Paths:
    embeddings_root: Path
    data_root: Path

    @classmethod
    def from_env(cls):
        return cls(
            embeddings_root=Path(os.getenv("EMBEDDINGS_ROOT")),
            data_root=Path(os.getenv("DATA_ROOT")),
        )
    
    # def run(self, id: str) -> 'RunPath':
    #     return RunPath(
    #         self.data_root / 'runs' / id,
    #     )
    
    def embedding(self, dataset: str, encoder: str) -> 'EmbeddingsPath':
        folder = f"cache-{dataset}-{encoder}"
        return EmbeddingsPath(
            self.embeddings_root / folder,
        )
    
# ============================================================================
# Embedding Paths
#
# Example use:
#   paths = Path.from_env()
#   k100k_train = paths.embedding("kather100k", "uni").split("train").load()
# ============================================================================

@dataclass
class EmbeddingsPath:
    root: Path

    def label_map(self) -> Path:
        return self.root / "label_map.json"

    def split(self, split: str) -> "EmbeddingsSplitPath":
        return EmbeddingsSplitPath(self.root / split)
    
    def load_splits(self) -> tuple["FeatureSpace", "FeatureSpace"]:
        from hproj.data.feature_space import get_train_test
        return get_train_test(self)


@dataclass
class EmbeddingsSplitPath:
    root: Path

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def emb_stats(self) -> Path:
        return self.root / "emb_stats.pt"

    def emb(self, shard: int | str) -> Path:
        return self.root / f"emb_{shard}.pt"

    def labels(self, shard: int | str) -> Path:
        return self.root / f"labels_{shard}.pt"

    def paths_json(self, shard: int | str) -> Path:
        return self.root / f"paths_{shard}.json"
    
    def glob_embs(self) -> list[Path]:
        pattern = "emb_[0-9][0-9][0-9][0-9].pt"
        return list(self.root.glob(pattern))

    def glob_labels(self) -> list[Path]:
        pattern = "labels_[0-9][0-9][0-9][0-9].pt"
        return list(self.root.glob(pattern))
    
    def glob_paths_json(self) -> list[Path]:
        pattern = "paths_[0-9][0-9][0-9][0-9].json"
        return list(self.root.glob(pattern))
    
    # non path related helper method
    def load(self) -> "FeatureSpace":
        from hproj.data.feature_space import load_embeddings
        return load_embeddings(self)


# ============================================================================
# Run Paths - for storing results of a run
# ============================================================================

@dataclass
class RunPath:
    root: Path

