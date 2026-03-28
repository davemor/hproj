from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import pandas as pd
from cuml.cluster import KMeans
from cuml.cluster.hdbscan import HDBSCAN, approximate_predict
from cuml.metrics.cluster import silhouette_score as cuml_silhouette_score
from cuml.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    pairwise_distances,
)
from sklearn.metrics import (
    silhouette_score as sk_silhouette_score,
)
from sklearn.preprocessing import normalize

from hproj.data.feature_space import FeatureSpace
from hproj.data.paths import Paths
from hproj.projectors.pca import PCAProjector
from hproj.util.atomic import atomic_write_csv, atomic_write_json
from hproj.util.config import Config
from hproj.util.logging import setup_logging
from hproj.util.seeds import set_seeds


@dataclass
class DatasetRepresentation:
    train: FeatureSpace
    test: FeatureSpace


@dataclass
class DatasetRepresentationSet:
    embeddings: DatasetRepresentation
    projections: dict[int, DatasetRepresentation]


def _to_numpy(array_like):
    if hasattr(array_like, "get"):
        return array_like.get()
    return np.asarray(array_like)


def _safe_silhouette_cuml(X, labels, metric: str) -> float:
    labels_np = _to_numpy(labels)
    if np.unique(labels_np).shape[0] < 2:
        return float("nan")
    return float(cuml_silhouette_score(X, labels, metric=metric))


def _safe_silhouette_sklearn(
    X_np: np.ndarray, labels_np: np.ndarray, metric: str
) -> float:
    if np.unique(labels_np).shape[0] < 2:
        return float("nan")
    return float(sk_silhouette_score(X_np, labels_np, metric=metric))


def build_representation_sets(
    paths: Paths,
    datasets: list[str],
    encoders: list[str],
    pca_dims: int,
    seed: int,
    logger,
) -> dict[tuple[str, str], DatasetRepresentationSet]:
    data: dict[tuple[str, str], DatasetRepresentationSet] = {}
    for encoder in encoders:
        for dataset in datasets:
            embeddings_path = paths.embedding(dataset, encoder)
            logger.info(
                "Loading embeddings for %s/%s from %s",
                dataset,
                encoder,
                embeddings_path.root,
            )
            train, test = embeddings_path.load_splits()

            projector = PCAProjector(n_components=pca_dims, seed=seed)
            projector.fit(train)
            train_proj = projector.transform(train)
            test_proj = projector.transform(test)

            data[(encoder, dataset)] = DatasetRepresentationSet(
                embeddings=DatasetRepresentation(train=train, test=test),
                projections={
                    pca_dims: DatasetRepresentation(train=train_proj, test=test_proj)
                },
            )
    return data


def evaluate_kmeans(
    data: dict[tuple[str, str], DatasetRepresentationSet],
    k_values: list[int],
    seed: int,
    logger,
) -> pd.DataFrame:
    rows = []
    for (encoder, dataset), rep_set in data.items():
        for rep_label, rep in [
            ("full", rep_set.embeddings),
            *rep_set.projections.items(),
        ]:
            X_train = rep.train.features
            X_test = rep.test.features
            y_test = rep.test.labels

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            sil_by_k = []
            for k in k_values:
                model = KMeans(n_clusters=k, random_state=seed, n_init=10)
                train_labels = model.fit_predict(X_train_scaled)
                sil_by_k.append(
                    _safe_silhouette_cuml(X_train_scaled, train_labels, metric="cosine")
                )

            sil_arr = np.asarray(sil_by_k, dtype=float)
            if np.all(np.isnan(sil_arr)):
                best_k = k_values[0]
            else:
                best_k = k_values[int(np.nanargmax(sil_arr))]

            best_model = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
            best_model.fit(X_train_scaled)
            test_labels = best_model.predict(X_test_scaled)

            y_test_np = _to_numpy(y_test)
            test_labels_np = _to_numpy(test_labels)
            row = {
                "encoder": encoder,
                "dataset": dataset,
                "representation": rep_label,
                "best_k": int(best_k),
                "silhouette": _safe_silhouette_cuml(
                    X_test_scaled, test_labels, metric="cosine"
                ),
                "ari": float(adjusted_rand_score(y_test_np, test_labels_np)),
                "ami": float(adjusted_mutual_info_score(y_test_np, test_labels_np)),
            }
            rows.append(row)
            logger.info(
                "kmeans %s/%s %s: k=%s sil=%.4f ari=%.4f ami=%.4f",
                encoder,
                dataset,
                rep_label,
                row["best_k"],
                row["silhouette"],
                row["ari"],
                row["ami"],
            )
    return pd.DataFrame(rows)


def _search_hdbscan(
    X_train_np: np.ndarray,
    min_cluster_sizes: list[int],
    min_samples_values: list[int | None],
) -> pd.DataFrame:
    rows = []
    for min_cluster_size in min_cluster_sizes:
        for min_samples in min_samples_values:
            model = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            train_labels = model.fit_predict(X_train_np)
            train_labels_np = _to_numpy(train_labels)
            train_mask = train_labels_np != -1

            n_clusters = (
                int(np.unique(train_labels_np[train_mask]).shape[0])
                if train_mask.any()
                else 0
            )
            noise_frac = float((~train_mask).mean())

            if train_mask.any():
                sil = _safe_silhouette_sklearn(
                    X_train_np[train_mask],
                    train_labels_np[train_mask],
                    metric="euclidean",
                )
            else:
                sil = float("nan")

            persistence = model.cluster_persistence_
            if persistence is not None and len(persistence) > 0:
                persistence_np = _to_numpy(persistence)
                mean_persistence = float(np.mean(persistence_np))
            else:
                mean_persistence = float("nan")

            rows.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    "train_silhouette": sil,
                    "n_clusters_train": n_clusters,
                    "noise_frac_train": noise_frac,
                    "mean_persistence": mean_persistence,
                }
            )

    return pd.DataFrame(rows)


def _select_best_hdbscan(
    search_df: pd.DataFrame,
    expected_cluster_min: int,
    expected_cluster_max: int,
) -> dict[str, int | None]:
    valid = search_df[
        (search_df["n_clusters_train"] >= 2)
        & (search_df["noise_frac_train"] <= 0.50)
        & (search_df["n_clusters_train"] <= 40)
    ].copy()

    if valid.empty:
        valid = search_df.sort_values(
            ["n_clusters_train", "noise_frac_train"], ascending=[False, True]
        ).head(1)

    expected_mid = (expected_cluster_min + expected_cluster_max) / 2
    valid["cluster_dist"] = (valid["n_clusters_train"] - expected_mid).abs()
    valid = valid.sort_values(
        ["train_silhouette", "cluster_dist", "noise_frac_train", "mean_persistence"],
        ascending=[False, True, True, False],
        na_position="last",
    )
    best = valid.iloc[0]
    best_min_samples = best["min_samples"]
    if pd.isna(best_min_samples):
        best_min_samples = None
    else:
        best_min_samples = int(best_min_samples)
    return {
        "min_cluster_size": int(best["min_cluster_size"]),
        "min_samples": best_min_samples,
    }


def evaluate_hdbscan(
    data: dict[tuple[str, str], DatasetRepresentationSet],
    min_cluster_sizes: list[int],
    min_samples_values: list[int | None],
    expected_cluster_min: int,
    expected_cluster_max: int,
    logger,
) -> pd.DataFrame:
    rows = []
    for (encoder, dataset), rep_set in data.items():
        for rep_label, rep in [
            ("full", rep_set.embeddings),
            *rep_set.projections.items(),
        ]:
            X_train_np = normalize(_to_numpy(rep.train.features), norm="l2")
            X_test_np = normalize(_to_numpy(rep.test.features), norm="l2")
            y_test_np = _to_numpy(rep.test.labels)

            search_df = _search_hdbscan(
                X_train_np, min_cluster_sizes, min_samples_values
            )
            best_params = _select_best_hdbscan(
                search_df,
                expected_cluster_min=expected_cluster_min,
                expected_cluster_max=expected_cluster_max,
            )

            model = HDBSCAN(
                min_cluster_size=best_params["min_cluster_size"],
                min_samples=best_params["min_samples"],
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=True,
            )
            train_labels = model.fit_predict(X_train_np)
            train_labels_np = _to_numpy(train_labels)
            train_mask = train_labels_np != -1

            test_labels, _ = approximate_predict(model, X_test_np)
            test_labels_np = _to_numpy(test_labels)
            test_mask = test_labels_np != -1

            n_clusters_train = (
                int(np.unique(train_labels_np[train_mask]).shape[0])
                if train_mask.any()
                else 0
            )
            n_clusters_test = (
                int(np.unique(test_labels_np[test_mask]).shape[0])
                if test_mask.any()
                else 0
            )

            if test_mask.any() and n_clusters_test >= 2:
                test_sil = _safe_silhouette_sklearn(
                    X_test_np[test_mask], test_labels_np[test_mask], metric="euclidean"
                )
            else:
                test_sil = float("nan")

            persistence = model.cluster_persistence_
            if persistence is not None and len(persistence) > 0:
                persistence_np = _to_numpy(persistence)
                mean_persistence = float(np.mean(persistence_np))
            else:
                mean_persistence = float("nan")

            row = {
                "encoder": encoder,
                "dataset": dataset,
                "representation": rep_label,
                "best_min_cluster_size": int(best_params["min_cluster_size"]),
                "best_min_samples": best_params["min_samples"],
                "n_clusters_train": n_clusters_train,
                "noise_frac_train": float((~train_mask).mean()),
                "n_clusters_test": n_clusters_test,
                "noise_frac_test": float((~test_mask).mean()),
                "mean_persistence_train": mean_persistence,
                "silhouette": test_sil,
                "ari": float(adjusted_rand_score(y_test_np, test_labels_np)),
                "ami": float(adjusted_mutual_info_score(y_test_np, test_labels_np)),
            }
            rows.append(row)
            logger.info(
                (
                    "hdbscan %s/%s %s: mcs=%s ms=%s "
                    "k_tr=%s k_te=%s sil=%.4f ari=%.4f ami=%.4f"
                ),
                encoder,
                dataset,
                rep_label,
                row["best_min_cluster_size"],
                row["best_min_samples"],
                row["n_clusters_train"],
                row["n_clusters_test"],
                row["silhouette"],
                row["ari"],
                row["ami"],
            )

    return pd.DataFrame(rows)


def _compute_rbf_gamma(X_train_np: np.ndarray) -> float:
    dists = pairwise_distances(X_train_np, metric="euclidean")
    upper = dists[np.triu_indices_from(dists, k=1)]
    median_dist = float(np.median(upper))
    if median_dist == 0:
        return 1.0
    return 1.0 / (2.0 * median_dist**2)


def evaluate_spectral(
    data: dict[tuple[str, str], DatasetRepresentationSet],
    k_values: list[int],
    affinities: list[str],
    nearest_neighbors: int,
    seed: int,
    logger,
) -> pd.DataFrame:
    rows = []
    for (encoder, dataset), rep_set in data.items():
        for rep_label, rep in [
            ("full", rep_set.embeddings),
            *rep_set.projections.items(),
        ]:
            X_train_np = normalize(_to_numpy(rep.train.features), norm="l2")
            X_test_np = normalize(_to_numpy(rep.test.features), norm="l2")
            y_test_np = _to_numpy(rep.test.labels)

            gamma = _compute_rbf_gamma(X_train_np)
            configs = []
            for k in k_values:
                if "rbf" in affinities:
                    configs.append((k, "rbf", {"gamma": gamma}))
                if "nearest_neighbors" in affinities:
                    configs.append(
                        (
                            k,
                            "nearest_neighbors",
                            {"n_neighbors": nearest_neighbors},
                        )
                    )

            best_cfg = None
            best_train_sil = float("nan")
            for k, affinity, affinity_kwargs in configs:
                model = SpectralClustering(
                    n_clusters=k,
                    affinity=affinity,
                    assign_labels="kmeans",
                    random_state=seed,
                    n_jobs=-1,
                    **affinity_kwargs,
                )
                train_labels = model.fit_predict(X_train_np)
                train_sil = _safe_silhouette_sklearn(
                    X_train_np, np.asarray(train_labels), metric="euclidean"
                )

                if np.isnan(train_sil):
                    continue
                if best_cfg is None or train_sil > best_train_sil:
                    best_cfg = (k, affinity, affinity_kwargs)
                    best_train_sil = train_sil

            if best_cfg is None:
                fallback_affinity = (
                    "rbf" if "rbf" in affinities else "nearest_neighbors"
                )
                fallback_kwargs = (
                    {"gamma": gamma}
                    if fallback_affinity == "rbf"
                    else {"n_neighbors": nearest_neighbors}
                )
                best_cfg = (k_values[0], fallback_affinity, fallback_kwargs)

            best_k, best_affinity, best_affinity_kwargs = best_cfg
            best_model = SpectralClustering(
                n_clusters=best_k,
                affinity=best_affinity,
                assign_labels="kmeans",
                random_state=seed,
                n_jobs=-1,
                **best_affinity_kwargs,
            )
            train_labels = np.asarray(best_model.fit_predict(X_train_np))

            unique_labels = np.unique(train_labels)
            centroids = np.vstack(
                [
                    X_train_np[train_labels == label].mean(axis=0)
                    for label in unique_labels
                ]
            )
            dists_to_centroids = pairwise_distances(
                X_test_np, centroids, metric="euclidean"
            )
            test_labels = unique_labels[np.argmin(dists_to_centroids, axis=1)]

            row = {
                "encoder": encoder,
                "dataset": dataset,
                "representation": rep_label,
                "best_k": int(best_k),
                "best_affinity": best_affinity,
                "silhouette": _safe_silhouette_sklearn(
                    X_test_np, np.asarray(test_labels), metric="euclidean"
                ),
                "ari": float(adjusted_rand_score(y_test_np, test_labels)),
                "ami": float(adjusted_mutual_info_score(y_test_np, test_labels)),
            }
            rows.append(row)
            logger.info(
                "spectral %s/%s %s: k=%s affinity=%s sil=%.4f ari=%.4f ami=%.4f",
                encoder,
                dataset,
                rep_label,
                row["best_k"],
                row["best_affinity"],
                row["silhouette"],
                row["ari"],
                row["ami"],
            )

    return pd.DataFrame(rows)


def _parse_min_samples(raw: str) -> list[int | None]:
    values: list[int | None] = []
    for token in (part.strip() for part in raw.split(",") if part.strip()):
        if token.lower() in {"none", "null"}:
            values.append(None)
        else:
            values.append(int(token))
    return values


@click.command()
@click.option(
    "--run-id",
    "-r",
    type=str,
    required=True,
    help="Run id to read datasets and encoders from.",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory to write clustering experiment outputs.",
)
@click.option(
    "--force", is_flag=True, help="Overwrite existing output files in output directory."
)
@click.option(
    "--seed",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for clustering runs.",
)
@click.option(
    "--pca-dims",
    type=int,
    default=9,
    show_default=True,
    help="PCA dimensions used for projected representation.",
)
@click.option(
    "--k-min",
    type=int,
    default=2,
    show_default=True,
    help="Minimum k to evaluate for KMeans/Spectral.",
)
@click.option(
    "--k-max",
    type=int,
    default=12,
    show_default=True,
    help="Maximum k to evaluate for KMeans/Spectral.",
)
@click.option(
    "--hdbscan-min-cluster-sizes",
    type=str,
    default="10,20,30",
    show_default=True,
    help="Comma-separated min_cluster_size grid for HDBSCAN.",
)
@click.option(
    "--hdbscan-min-samples",
    type=str,
    default="none,10,20",
    show_default=True,
    help="Comma-separated min_samples grid for HDBSCAN. Use 'none' to include None.",
)
@click.option(
    "--expected-cluster-min",
    type=int,
    default=9,
    show_default=True,
    help="Lower bound of expected cluster count for HDBSCAN model selection.",
)
@click.option(
    "--expected-cluster-max",
    type=int,
    default=24,
    show_default=True,
    help="Upper bound of expected cluster count for HDBSCAN model selection.",
)
@click.option(
    "--spectral-affinity",
    "spectral_affinities",
    type=click.Choice(["rbf", "nearest_neighbors"], case_sensitive=False),
    multiple=True,
    default=("rbf", "nearest_neighbors"),
    show_default=True,
    help=(
        "Affinity modes to search for spectral clustering; "
        "repeat flag to select subset."
    ),
)
@click.option(
    "--spectral-neighbors",
    type=int,
    default=15,
    show_default=True,
    help="n_neighbors used by nearest-neighbors spectral affinity.",
)
@click.option(
    "--notebook-parity",
    is_flag=True,
    help=(
        "Override slim defaults with notebook search spaces "
        "(k=2..32, full HDBSCAN grids)."
    ),
)
def clustering(
    run_id: str,
    output_dir: Path,
    force: bool,
    seed: int,
    pca_dims: int,
    k_min: int,
    k_max: int,
    hdbscan_min_cluster_sizes: str,
    hdbscan_min_samples: str,
    expected_cluster_min: int,
    expected_cluster_max: int,
    spectral_affinities: tuple[str, ...],
    spectral_neighbors: int,
    notebook_parity: bool,
):
    """Run clustering experiment and write outputs to a chosen directory."""
    if pca_dims < 1:
        raise click.UsageError("--pca-dims must be >= 1")
    if k_min < 2 or k_max < 2 or k_min > k_max:
        raise click.UsageError("Require 2 <= --k-min <= --k-max")
    if expected_cluster_min > expected_cluster_max:
        raise click.UsageError(
            "--expected-cluster-min cannot exceed --expected-cluster-max"
        )
    if spectral_neighbors < 2:
        raise click.UsageError("--spectral-neighbors must be >= 2")

    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    if not run_paths.config().exists():
        raise click.UsageError(
            f"No config found for run id '{run_id}' at {run_paths.config()}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "clustering.log"
    logger = setup_logging(log_file)

    output_files = {
        "kmeans": output_dir / "kmeans_results.csv",
        "hdbscan": output_dir / "hdbscan_results.csv",
        "spectral": output_dir / "spectral_results.csv",
        "metadata": output_dir / "metadata.json",
    }
    existing = [path for path in output_files.values() if path.exists()]
    if existing and not force:
        existing_str = ", ".join(str(path) for path in existing)
        raise click.UsageError(
            f"Output files already exist ({existing_str}). Use --force to overwrite."
        )

    if notebook_parity:
        k_values = list(range(2, 33))
        min_cluster_sizes = [5, 10, 15, 20, 30, 40, 60]
        min_samples_values = [None, 5, 10, 15, 20]
    else:
        k_values = list(range(k_min, k_max + 1))
        min_cluster_sizes = [
            int(token.strip())
            for token in hdbscan_min_cluster_sizes.split(",")
            if token.strip()
        ]
        min_samples_values = _parse_min_samples(hdbscan_min_samples)

    if not min_cluster_sizes:
        raise click.UsageError("--hdbscan-min-cluster-sizes produced an empty grid")
    if not min_samples_values:
        raise click.UsageError("--hdbscan-min-samples produced an empty grid")

    cfg = Config.from_yaml(run_paths.config())
    if not cfg.datasets or not cfg.encoders:
        raise click.UsageError(
            "Run config must contain non-empty datasets and encoders"
        )

    set_seeds(seed)
    logger.info("Running clustering experiment for run id %s", run_id)
    logger.info("Writing outputs to %s", output_dir)

    start_time = perf_counter()
    data = build_representation_sets(
        paths=paths,
        datasets=list(cfg.datasets),
        encoders=list(cfg.encoders),
        pca_dims=pca_dims,
        seed=seed,
        logger=logger,
    )

    kmeans_results = evaluate_kmeans(
        data=data, k_values=k_values, seed=seed, logger=logger
    )
    hdbscan_results = evaluate_hdbscan(
        data=data,
        min_cluster_sizes=min_cluster_sizes,
        min_samples_values=min_samples_values,
        expected_cluster_min=expected_cluster_min,
        expected_cluster_max=expected_cluster_max,
        logger=logger,
    )
    spectral_results = evaluate_spectral(
        data=data,
        k_values=k_values,
        affinities=[affinity.lower() for affinity in spectral_affinities],
        nearest_neighbors=spectral_neighbors,
        seed=seed,
        logger=logger,
    )

    atomic_write_csv(output_files["kmeans"], kmeans_results)
    atomic_write_csv(output_files["hdbscan"], hdbscan_results)
    atomic_write_csv(output_files["spectral"], spectral_results)

    metadata = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "output_dir": str(output_dir),
        "datasets": list(cfg.datasets),
        "encoders": list(cfg.encoders),
        "seed": seed,
        "pca_dims": pca_dims,
        "notebook_parity": notebook_parity,
        "k_values": k_values,
        "hdbscan_min_cluster_sizes": min_cluster_sizes,
        "hdbscan_min_samples": min_samples_values,
        "expected_cluster_range": [expected_cluster_min, expected_cluster_max],
        "spectral_affinities": [affinity.lower() for affinity in spectral_affinities],
        "spectral_neighbors": spectral_neighbors,
        "rows": {
            "kmeans": int(kmeans_results.shape[0]),
            "hdbscan": int(hdbscan_results.shape[0]),
            "spectral": int(spectral_results.shape[0]),
        },
        "elapsed_seconds": perf_counter() - start_time,
    }
    atomic_write_json(output_files["metadata"], metadata)

    logger.info("Wrote %s", output_files["kmeans"])
    logger.info("Wrote %s", output_files["hdbscan"])
    logger.info("Wrote %s", output_files["spectral"])
    logger.info("Wrote %s", output_files["metadata"])
