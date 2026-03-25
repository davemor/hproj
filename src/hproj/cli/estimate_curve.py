import json
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

import click
import matplotlib.pyplot as plt
import pandas as pd
from dask.distributed import as_completed
from tqdm import tqdm

from hproj.classifiers.classifier import ClassifierFactory
from hproj.classifiers.metrics import compute_classification_metrics
from hproj.cli.calibrate_classifier import load_task_results
from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.measure.measurement import MeasurementFactory
from hproj.projectors.projector import ProjectorFactory
from hproj.util.atomic import atomic_write_json
from hproj.util.config import ClassifierConfig, Config, ProjectorConfig
from hproj.util.dask import make_dask_client
from hproj.util.logging import setup_logging
from hproj.util.seeds import make_seed


@dataclass
class TaskDescription:
    projector_cfg: ProjectorConfig
    classifier_cfg: ClassifierConfig
    dataset_name: str
    n_components: int
    base_seed: int

    def key(self) -> str:
        return (
            f"{self.projector_cfg.name}_{self.classifier_cfg.name}_"
            f"{self.dataset_name}_{self.n_components}_{self.base_seed}"
        )


def load_best_params(path, logger, name):
    if path.exists():
        with open(path) as f:
            return json.load(f)

    logger.info(f"Skipping loading hyperparameters for {name}.")
    return {}


def build_projector_configs(cfg, paths, run_id, logger):
    return [
        ProjectorConfig(
            name=name,
            params=load_best_params(
                paths.run(run_id).calibration().projector(name).best_params(),
                logger,
                name,
            ),
        )
        for name in cfg.curve.projectors
    ]


def build_classifier_configs(cfg, paths, run_id, logger):
    return [
        ClassifierConfig(
            name=name,
            params=load_best_params(
                paths.run(run_id).calibration().classifier(name).best_params(),
                logger,
                name,
            ),
        )
        for name in cfg.curve.classifiers
    ]


def plot_metric_with_error_bars(
    df,
    metric: str,
    output_path,
    x_col: str = "n_components",
    group_col: str = "projector",
    y_col: str | None = None,
    err_col: str | None = None,
    title: str | None = None,
):
    y_col = y_col or f"{metric}_seed_mean_dataset_mean"
    err_col = err_col or f"{metric}_seed_mean_dataset_std"
    output_path = Path(output_path)

    required_cols = {x_col, group_col, y_col, err_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(8, 5))

    for group_value, group_df in df.groupby(group_col):
        group_df = group_df.sort_values(x_col)
        ax.errorbar(
            group_df[x_col],
            group_df[y_col],
            yerr=group_df[err_col],
            marker="o",
            capsize=4,
            label=str(group_value),
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs {x_col}")
    ax.legend(title=group_col)
    ax.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return output_path


def score_curve_task(embeddings, task_desc, folds, measurement_cfgs):
    seed = make_seed(
        task_desc.base_seed,
        task_desc.projector_cfg.name,
        task_desc.classifier_cfg.name,
        task_desc.dataset_name,
        task_desc.n_components,
    )

    projector = ProjectorFactory.create(
        task_desc.projector_cfg.name,
        task_desc.n_components,
        seed,
        **task_desc.projector_cfg.params,
    )
    classifier = ClassifierFactory.create(
        task_desc.classifier_cfg.name,
        seed,
        **task_desc.classifier_cfg.params,
    )
    measurements = [
        (cfg.name, MeasurementFactory.create(cfg.name, seed, **cfg.params))
        for cfg in measurement_cfgs
    ]

    mean_scores = score_feature_space(embeddings, folds, classifier, measurements, projector)

    metadata = {
        "classifier": task_desc.classifier_cfg.name,
        "dataset": task_desc.dataset_name,
        "projector": task_desc.projector_cfg.name,
        "n_components": task_desc.n_components,
        "base_seed": task_desc.base_seed,
        "seed": seed,
    }

    return metadata | mean_scores

def score_feature_space(embeddings, folds, classifier, measurements, projector = None):
    fold_scores = defaultdict(list)

    for train, valid in embeddings.get_folds(folds):
        if projector:
            projector.fit(train)
            train_proj = projector.transform(train)
            valid_proj = projector.transform(valid)
        else:
            train_proj = train
            valid_proj = valid

        for name, measurement in measurements:
            fold_scores[name].append(
                float(measurement(train_proj, valid_proj, train, valid))
            )

        classifier.fit(train_proj)
        y_pred, y_scores = classifier.predict_and_score(valid_proj)

        for name, value in compute_classification_metrics(
            valid.labels, y_pred, y_scores
        ).items():
            fold_scores[name].append(float(value))

    mean_scores = {
        name: float(mean(values))
        for name, values in fold_scores.items()
        if values
    }
    
    return mean_scores

def score_unprojected_embeddings(embeddings, base_seed, classifier_cfg, dataset_name, folds, measurement_cfgs):
    seed = make_seed(
        base_seed,
        "noprojector",
        classifier_cfg.name,
        dataset_name,
        f"{embeddings.num_dimensions()}",
    )

    classifier = ClassifierFactory.create(
        classifier_cfg.name,
        seed,
        **classifier_cfg.params,
    )
    measurements = [
        (cfg.name, MeasurementFactory.create(cfg.name, seed, **cfg.params))
        for cfg in measurement_cfgs
    ]

    mean_scores = score_feature_space(embeddings, folds, classifier, measurements)

    metadata = {
        "classifier": classifier_cfg.name,
        "dataset": dataset_name,
        "n_components": embeddings.num_dimensions(),
        "base_seed": base_seed,
        "seed": seed,
    }

    return metadata | mean_scores

def aggregate_group(df, cols_group, spec):
    stats = df.groupby(cols_group, dropna=False)
    stats = stats.agg(**spec).reset_index()
    stats = stats.fillna(0.0)
    return stats


def aggregate_curve_results_across_seeds(results_df):
    group_cols = ["classifier", "dataset", "projector", "n_components"]

    value_cols = [
        col
        for col in results_df.columns
        if col not in {*group_cols, "base_seed", "seed"}
    ]

    spec = {
        f"{col}_seed_mean": (col, "mean")
        for col in value_cols
    } | {
        f"{col}_seed_std": (col, "std")
        for col in value_cols
    }

    return aggregate_group(results_df, group_cols, spec)


def aggregate_across_datasets(df_seed):
    group_cols = ["classifier", "projector", "n_components"]

    value_cols = [
        col
        for col in df_seed.columns
        if col not in {*group_cols, "dataset"}
    ]

    spec = {
        f"{col}_dataset_mean": (col, "mean")
        for col in value_cols
    } | {
        f"{col}_dataset_std": (col, "std")
        for col in value_cols
    }

    return aggregate_group(df_seed, group_cols, spec)


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
@click.option("--force", is_flag=True, help="Recompute existing task outputs.")
def estimate_curve(run_id: str, force: bool):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    curve_paths = run_paths.curve()
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Running the curve estimation step of the experiment.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    train_embeddings = {
        f"{dataset}_{encoder}": paths.embedding(dataset, encoder).split("train").load()
        for dataset, encoder in product(cfg.datasets, cfg.encoders)
    }

    if cfg.curve.subsample:
        train_embeddings = {
            name: embeddings.stratified_sample(cfg.curve.subsample)
            for name, embeddings in train_embeddings.items()
        }

    dataset_folds = {
        name: generate_stratified_folds(embeddings, n_splits=cfg.num_folds, seed=42)
        for name, embeddings in train_embeddings.items()
    }

    projector_cfgs = build_projector_configs(cfg, paths, run_id, logger)
    classifier_cfgs = build_classifier_configs(cfg, paths, run_id, logger)

    curve_paths.mkdir()
    curve_paths.summaries().mkdir()
    curve_paths.plots().mkdir()

    client, cluster = make_dask_client()

    try:
        start_time = perf_counter()

        for classifier_cfg in classifier_cfgs:
            classifier_results = []

            for projector_cfg in projector_cfgs:
                output_path = curve_paths.projector_classifier(
                    projector_cfg.name,
                    classifier_cfg.name,
                )
                output_path.mkdir()
                tasks_dir = output_path.tasks()
                tasks_dir.mkdir()

                unprojected_scores_list = []

                for dataset_name, embeddings in train_embeddings.items():
                    folds = dataset_folds[dataset_name]
                    embeddings_future = client.scatter(embeddings, broadcast=True)
                    desc = f"{classifier_cfg.name} | {projector_cfg.name} | {dataset_name}"

                    # optionally - assess at the full number of dimensions (but projected)
                    dimensions = cfg.curve.dimensions
                    if cfg.curve.include_full_dimension:
                        dimensions.append(embeddings.num_dimensions())

                    pending_tasks = {}

                    for n_components in dimensions:
                        for base_seed in cfg.seeds.curve:
                            unprojected_scores = score_unprojected_embeddings(embeddings, base_seed, classifier_cfg, dataset_name, folds, cfg.curve.measurements)
                            unprojected_scores_list.append(unprojected_scores)

                            task_desc = TaskDescription(
                                projector_cfg=projector_cfg,
                                classifier_cfg=classifier_cfg,
                                dataset_name=dataset_name,
                                n_components=n_components,
                                base_seed=base_seed,
                            )
                            task_file = tasks_dir.task(task_desc.key())

                            if task_file.exists() and not force:
                                continue

                            future = client.submit(
                                score_curve_task,
                                embeddings_future,
                                task_desc,
                                folds,
                                cfg.curve.measurements,
                                pure=False,
                            )
                            pending_tasks[future] = task_file

                    logger.info(f"{desc}: {len(pending_tasks)} pending task(s)")

                    for future in tqdm(
                        as_completed(pending_tasks),
                        total=len(pending_tasks),
                        desc=desc,
                    ):
                        atomic_write_json(pending_tasks[future], future.result())

                unprojected_scores_df = pd.DataFrame(unprojected_scores_list)
                unprojected_scores_df.to_csv(output_path.unprojected_scores())

                results_df = load_task_results(output_path.tasks())
                classifier_results.append(results_df)

                results_df.to_csv(output_path.results(), index=False)
                logger.info(f"Writing results csv to {output_path.results()}")

                seed_summary_df = aggregate_curve_results_across_seeds(results_df)
                seed_summary_df.to_csv(output_path.seed_summary_results(), index=False)
                logger.info(
                    f"Writing summary results csv to {output_path.seed_summary_results()}"
                )

                dataset_summary_df = aggregate_across_datasets(seed_summary_df)
                dataset_summary_df.to_csv(output_path.dataset_summary_results(), index=False)
                logger.info(
                    f"Writing summary results csv to {output_path.dataset_summary_results()}"
                )

            classifier_results_df = pd.concat(classifier_results, ignore_index=True)

            classifier_seed_summary_df = aggregate_curve_results_across_seeds(
                classifier_results_df
            )
            classifier_dataset_summary_df = aggregate_across_datasets(
                classifier_seed_summary_df
            )

            classifier_seed_summary_path = curve_paths.summaries().classifier_seed_summary(
                classifier_cfg.name
            )
            classifier_dataset_summary_path = (
                curve_paths.summaries().classifier_dataset_summary(classifier_cfg.name)
            )

            classifier_seed_summary_df.to_csv(classifier_seed_summary_path, index=False)
            classifier_dataset_summary_df.to_csv(
                classifier_dataset_summary_path, index=False
            )

            logger.info(f"Writing classifier seed summary to {classifier_seed_summary_path}")
            logger.info(
                f"Writing classifier dataset summary to {classifier_dataset_summary_path}"
            )

            accuracy_plot_path = curve_paths.plots().classifier_accuracy(
                classifier_cfg.name
            )
            roc_auc_plot_path = curve_paths.plots().classifier_roc_auc(
                classifier_cfg.name
            )

            plot_metric_with_error_bars(
                df=classifier_dataset_summary_df,
                metric="accuracy",
                output_path=accuracy_plot_path,
                title=f"{classifier_cfg.name}: accuracy vs n_components",
            )
            logger.info(f"Wrote accuracy plot to {accuracy_plot_path}")

            plot_metric_with_error_bars(
                df=classifier_dataset_summary_df,
                metric="roc_auc",
                output_path=roc_auc_plot_path,
                title=f"{classifier_cfg.name}: roc_auc vs n_components",
            )
            logger.info(f"Wrote roc_auc plot to {roc_auc_plot_path}")

        elapsed_time = perf_counter() - start_time
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Run output saved to: {run_paths.root}")

    finally:
        client.close()
        cluster.close()