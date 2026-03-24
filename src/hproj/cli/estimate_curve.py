from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
from statistics import mean
from time import perf_counter

import click
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
    configs = []
    for name in cfg.curve.projectors:
        params = load_best_params(
            paths.run(run_id).calibration().projector(name).best_params(),
            logger,
            name,
        )
        configs.append(ProjectorConfig(name=name, params=params))
    return configs


def build_classifier_configs(cfg, paths, run_id, logger):
    configs = []
    for name in cfg.curve.classifiers:
        params = load_best_params(
            paths.run(run_id).calibration().classifier(name).best_params(),
            logger,
            name,
        )
        configs.append(ClassifierConfig(name=name, params=params))
    return configs


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

    fold_scores = defaultdict(list)

    for train, valid in embeddings.get_folds(folds):
        projector.fit(train)
        train_proj = projector.transform(train)
        valid_proj = projector.transform(valid)

        for name, measurement in measurements:
            fold_scores[name].append(float(measurement(train_proj, valid_proj, train, valid)))

        classifier.fit(train_proj)
        y_pred, y_scores = classifier.predict_and_score(valid_proj)

        for name, value in compute_classification_metrics(valid.labels, y_pred, y_scores).items():
            fold_scores[name].append(float(value))

    mean_scores = {
        name: float(mean(values))
        for name, values in fold_scores.items()
        if values
    }

    metadata = {
        "classifier": task_desc.classifier_cfg.name,
        "dataset": task_desc.dataset_name,
        "projector": task_desc.projector_cfg.name,
        "n_components": task_desc.n_components,
        "base_seed": task_desc.base_seed,
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

@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
@click.option("--force", is_flag=True, help="Recompute existing task outputs.")
def estimate_curve(run_id: str, force: bool):
    paths = Paths.from_env()
    cfg = Config.from_yaml(paths.run(run_id).config())

    logger = setup_logging(paths.run(run_id).log_file())
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

    client, cluster = make_dask_client()

    try:
        start_time = perf_counter()

        for projector_cfg in projector_cfgs:
            for classifier_cfg in classifier_cfgs:
                output_path = paths.run(run_id).curve().projector_classifier(
                    projector_cfg.name,
                    classifier_cfg.name,
                )
                output_path.mkdir()
                tasks_dir = output_path.tasks()
                tasks_dir.mkdir()

                for dataset_name, embeddings in train_embeddings.items():
                    folds = dataset_folds[dataset_name]
                    embeddings_future = client.scatter(embeddings, broadcast=True)
                    desc = f"{classifier_cfg.name} | {projector_cfg.name} | {dataset_name}"

                    pending_tasks = {}
                    for n_components in cfg.curve.dimensions:
                        for base_seed in cfg.seeds.curve:
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

                results_df = load_task_results(output_path.tasks())
                results_df.to_csv(output_path.results())
                logger.info(f"Writing results csv to {output_path.results()}")

                summary_results_df = aggregate_curve_results_across_seeds(results_df)
                summary_results_df.to_csv(output_path.results())
                logger.info(f"Writing summary results csv to {output_path.summary_results()}")   

        elapsed_time = perf_counter() - start_time
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Run output saved to: {paths.run(run_id).root}")

    finally:
        client.close()
        cluster.close()