import json
from dataclasses import dataclass
from itertools import product
from statistics import mean
from time import perf_counter

import click
import pandas as pd
from dask.distributed import as_completed
from tqdm.auto import tqdm

from hproj.classifiers.classifier import ClassifierFactory
from hproj.classifiers.metrics import compute_classification_metrics
from hproj.data.folds import Fold, generate_stratified_folds
from hproj.data.paths import Paths
from hproj.projectors.projector import ProjectorFactory
from hproj.util.atomic import atomic_write_json
from hproj.util.config import Config
from hproj.util.dask import make_dask_client
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging
from hproj.util.seeds import make_seed

METRICS = ["accuracy", "roc_auc"]


@dataclass
class TaskDescription:
    classifier_name: str
    classifier_params: dict
    dataset: str
    n_components: int
    base_seed: int
    projector_name: str
    projector_params: dict
    params_idx: int

    def key(self):
        return (
            f"{self.classifier_name}_{self.dataset}_d{self.n_components}"
            f"_hp{self.params_idx}_s{self.base_seed}_p{self.projector_name}"
        )


def score_classifier_config(
    embeddings,
    projector_name: str,
    projector_params: dict,
    classifier_name: str,
    classifier_params: dict,
    n_components: int,
    seed: int,
    folds: list[Fold],
) -> dict:
    """
    Score a classifier hyperparameter configuration using fixed projector
    hyperparameters. Each metric is averaged across folds.
    """
    fold_scores = {metric: [] for metric in METRICS}

    for train, valid in embeddings.get_folds(folds):
        projector = ProjectorFactory.create(
            projector_name,
            n_components,
            seed,
            **projector_params,
        )
        projector.fit(train)
        train_proj = projector.transform(train)
        valid_proj = projector.transform(valid)

        classifier = ClassifierFactory.create(
            classifier_name,
            seed=seed,
            **classifier_params,
        )
        classifier.fit(train_proj)
        y_pred, y_scores = classifier.predict_and_score(valid_proj)

        scores = compute_classification_metrics(valid.labels, y_pred, y_scores)
        for metric_name, score in scores.items():
            fold_scores[metric_name].append(float(score))

    return {
        metric_name: float(mean(scores))
        for metric_name, scores in fold_scores.items()
    }


def score_classifier_task(
    embeddings_future,
    task_desc: TaskDescription,
    folds: list[Fold],
):
    seed = make_seed(
        task_desc.base_seed,
        task_desc.dataset,
        task_desc.projector_name,
        task_desc.projector_params,
        task_desc.classifier_name,
        task_desc.classifier_params,
        task_desc.n_components,
    )

    scores = score_classifier_config(
        embeddings=embeddings_future,
        projector_name=task_desc.projector_name,
        projector_params=task_desc.projector_params,
        classifier_name=task_desc.classifier_name,
        classifier_params=task_desc.classifier_params,
        n_components=task_desc.n_components,
        seed=seed,
        folds=folds,
    )

    meta_data = {
        "classifier": task_desc.classifier_name,
        "dataset": task_desc.dataset,
        "projector": task_desc.projector_name,
        "n_components": task_desc.n_components,
        "base_seed": task_desc.base_seed,
        "seed": seed,
        "params_idx": task_desc.params_idx,
    }

    return meta_data | task_desc.classifier_params | scores


def load_task_results(tasks_dir):
    def load_task_result(task_path):
        with open(task_path) as f:
            return json.load(f)

    task_paths = tasks_dir.glob_tasks()
    results = [load_task_result(p) for p in task_paths]
    return pd.DataFrame(results)


def summaries_results_for_classifier(
    results_df: pd.DataFrame,
    rank_metric: str,
    sort_desc: bool = True,
) -> pd.DataFrame:
    """
    Aggregate:
      mean/std across seeds
      median across dimensions
      mean/std across datasets
      mean/std across projectors

    This produces one ranking per params_idx across all projectors.
    """
    assert rank_metric in METRICS, f"rank_metric must be one of {METRICS}"

    def aggregate_group(df, cols_group, spec):
        stats = df.groupby(cols_group, dropna=False)
        stats = stats.agg(**spec).reset_index()
        stats = stats.fillna(0.0)
        return stats

    def aggregate_across_seeds():
        cols_group = ["projector", "dataset", "params_idx", "n_components"]
        spec = {}
        for metric in METRICS:
            spec[f"{metric}_seed_mean"] = (metric, "mean")
            spec[f"{metric}_seed_std"] = (metric, "std")
        return aggregate_group(results_df, cols_group, spec)

    def aggregate_across_dimensions(seed_stats: pd.DataFrame):
        cols_group = ["projector", "dataset", "params_idx"]
        spec = {}
        for metric in METRICS:
            spec[f"{metric}_dim_median"] = (f"{metric}_seed_mean", "median")
            spec[f"{metric}_dim_std"] = (f"{metric}_seed_mean", "std")
            spec[f"{metric}_seed_std_mean"] = (f"{metric}_seed_std", "mean")
        return aggregate_group(seed_stats, cols_group, spec)

    def aggregate_across_datasets(dim_stats: pd.DataFrame):
        cols_group = ["projector", "params_idx"]
        spec = {}
        for metric in METRICS:
            spec[f"{metric}_dataset_mean"] = (f"{metric}_dim_median", "mean")
            spec[f"{metric}_dataset_std"] = (f"{metric}_dim_median", "std")
            spec[f"{metric}_dim_std_mean"] = (f"{metric}_dim_std", "mean")
            spec[f"{metric}_seed_std_mean"] = (f"{metric}_seed_std_mean", "mean")
        return aggregate_group(dim_stats, cols_group, spec)

    def aggregate_across_projectors(dataset_stats: pd.DataFrame):
        cols_group = ["params_idx"]
        spec = {}
        for metric in METRICS:
            spec[f"{metric}_mean"] = (f"{metric}_dataset_mean", "mean")
            spec[f"{metric}_projector_std"] = (f"{metric}_dataset_mean", "std")
            spec[f"{metric}_dataset_std_mean"] = (f"{metric}_dataset_std", "mean")
            spec[f"{metric}_dim_std_mean"] = (f"{metric}_dim_std_mean", "mean")
            spec[f"{metric}_seed_std_mean"] = (f"{metric}_seed_std_mean", "mean")
        return aggregate_group(dataset_stats, cols_group, spec)

    seed_stats = aggregate_across_seeds()
    dim_stats = aggregate_across_dimensions(seed_stats)
    dataset_stats = aggregate_across_datasets(dim_stats)
    projector_stats = aggregate_across_projectors(dataset_stats)

    ranked = projector_stats.sort_values(
        by=[
            f"{rank_metric}_mean",
            f"{rank_metric}_projector_std",
            f"{rank_metric}_dataset_std_mean",
            f"{rank_metric}_seed_std_mean",
            f"{rank_metric}_dim_std_mean",
        ],
        ascending=[not sort_desc, True, True, True, True],
        kind="mergesort",
    )
    return ranked


def attach_classifier_params_from_grid(
    ranked_df: pd.DataFrame,
    param_grid: list[dict],
    params_col: str = "params_idx",
) -> pd.DataFrame:
    out = ranked_df.copy()
    param_dicts = out[params_col].map(lambda i: param_grid[i])
    params_expanded = pd.DataFrame(param_dicts.tolist(), index=out.index)
    return pd.concat([out, params_expanded], axis=1)


@click.command()
@click.option(
    "--run-id",
    "-r",
    type=str,
    help="The id of a run to continue.",
    required=True,
)
@click.option("--force", is_flag=True, help="Recompute existing task outputs.")
def calibrate_classifier(run_id: str, force: bool):
    paths = Paths.from_env()
    cfg = Config.from_yaml(paths.run(run_id).config())

    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running the classifier calibration step of the experiment.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    train_embeddings = {
        f"{dataset}_{encoder}": paths.embedding(dataset, encoder).split("train").load()
        for dataset, encoder in product(cfg.datasets, cfg.encoders)
    }

    num_subsamples = cfg.calibration.classifier.subsample
    if num_subsamples:
        train_embeddings = {
            name: embeddings.stratified_sample(num_subsamples)
            for name, embeddings in train_embeddings.items()
        }

    num_folds = cfg.num_folds
    dataset_folds = {
        name: generate_stratified_folds(ds, n_splits=num_folds, seed=42)
        for name, ds in train_embeddings.items()
    }

    client, cluster = make_dask_client()

    projector_cfgs = []
    for projector in cfg.calibration.projector.projectors:
        if len(projector.params) == 0:
            projector_cfgs.append((projector.name, {}))
            continue

        best_params_path = (
            paths.run(run_id)
            .calibration()
            .projector(projector.name)
            .best_params()
        )
        with open(best_params_path) as f:
            best_params = json.load(f)
        projector_cfgs.append((projector.name, best_params))

    try:
        start_time = perf_counter()

        for classifier in cfg.calibration.classifier.classifiers:
            logger.info(f"Evaluating classifier: {classifier.name}")
            logger.info(f"Hyperparams: {classifier.params}")
            if len(classifier.params) == 0:
                logger.info("No hyperparameters to tune, skipping.")
                continue

            output_path = paths.run(run_id).calibration().classifier(classifier.name)
            output_path.mkdir()
            tasks_dir = output_path.tasks()
            tasks_dir.mkdir()

            param_grid = make_param_grid(classifier.params)
            logger.info(f"Generated {len(param_grid)} hyperparameter configs.")

            for projector_name, projector_config in projector_cfgs:
                logger.info(f"Using fixed projector config for {projector_name}: {projector_config}")

                for dataset_name, embeddings in train_embeddings.items():
                    logger.info(
                        f"Evaluating classifier {classifier.name} with projector "
                        f"{projector_name} on dataset: {dataset_name}"
                    )
                    logger.info(f"Number of samples: {embeddings.num_samples()}")

                    embeddings_future = client.scatter(embeddings, broadcast=True)
                    folds = dataset_folds[dataset_name]
                    base_seeds = cfg.seeds.calibration.classifier

                    for n_components in cfg.calibration.classifier.dimensions:
                        pending_tasks = {}

                        for params_idx, classifier_params in enumerate(param_grid):
                            for base_seed in base_seeds:
                                task_desc = TaskDescription(
                                    classifier_name=classifier.name,
                                    classifier_params=classifier_params,
                                    dataset=dataset_name,
                                    n_components=n_components,
                                    base_seed=base_seed,
                                    projector_name=projector_name,
                                    projector_params=projector_config,
                                    params_idx=params_idx,
                                )

                                task_file = tasks_dir.task(task_desc.key())
                                if task_file.exists() and not force:
                                    continue

                                future = client.submit(
                                    score_classifier_task,
                                    embeddings_future,
                                    task_desc,
                                    folds,
                                    pure=False,
                                )
                                pending_tasks[future] = task_file

                        desc = (
                            f"{classifier.name} | {projector_name} | "
                            f"{dataset_name} | d={n_components}"
                        )
                        for future in tqdm(
                            as_completed(pending_tasks.keys()),
                            total=len(pending_tasks),
                            desc=desc,
                        ):
                            results = future.result()
                            task_file = pending_tasks[future]
                            atomic_write_json(task_file, results)

            results_df = load_task_results(output_path.tasks())
            summary_df = summaries_results_for_classifier(
                results_df,
                rank_metric=cfg.calibration.classifier.select,
            )
            summary_df = attach_classifier_params_from_grid(summary_df, param_grid)

            logger.info("")
            logger.info(f"Classifier Config Summary for {classifier.name}")
            logger.info(f"\n{summary_df}")

            logger.info(f"Saving results to {output_path.root}")
            results_df.to_csv(output_path.scores(), index=False)
            summary_df.to_csv(output_path.summary(), index=False)

            # Save one global best config across all projectors
            best_row = summary_df.iloc[0]
            best_param_idx = int(best_row.params_idx)
            best_params = param_grid[best_param_idx]

            best_params_path = (
                paths.run(run_id)
                .calibration()
                .classifier(classifier.name)
                .best_params()
            )
            atomic_write_json(best_params_path, best_params)

            logger.info(
                f"The best global hyperparameters for classifier {classifier.name} were:"
            )
            logger.info(best_params)

        elapsed_time = perf_counter() - start_time
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Run output saved to: {paths.run(run_id).root}")

    finally:
        client.close()
        cluster.close()