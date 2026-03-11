from itertools import product
import json
from pathlib import Path
from statistics import mean
from time import perf_counter

import click
import numpy as np
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import pandas as pd

from dask.distributed import as_completed
from tqdm.auto import tqdm

from hproj.data.feature_space import FeatureSpace
from hproj.data.folds import Fold, generate_stratified_folds
from hproj.data.paths import Paths
from hproj.measure.measurement import Measurement, MeasurementFactory
from hproj.projectors.projector import Projector, ProjectorFactory
from hproj.util.config import Config, MeasurementConfig
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging
from hproj.util.runs import generate_run_id


def score_projector_config(
    embeddings: FeatureSpace,
    projector: Projector,
    measurements: list[Measurement],
    folds: list[Fold],
) -> dict:
    # this function scores a projector (that has been configured with some hyperparameters)
    # by applying it to the given embeddings and measuring the quality of the resulting projection with the given measurement.
    # The score is averaged across the given folds.
    fold_scores = {measurement.name: [] for measurement in measurements}
    for train, valid in embeddings.get_folds(folds):
        projector.fit(train)
        train_proj = projector.transform(train)
        valid_proj = projector.transform(valid)
        for measurement in measurements:
            key = measurement.name
            score = measurement(train_proj, valid_proj, train, valid)
            fold_scores[key].append(score)

    # compute the mean for each measurement across the folds
    mean_scores = {
        key: float(mean(fold_scores)) for key, fold_scores in fold_scores.items()
    }
    return mean_scores


def score_projector_config_task(
    embeddings_future,
    dataset_name: str,
    proj_name: str,
    proj_hyperparams: dict,
    measurement_configs: list[MeasurementConfig],
    n_components,
    seeds,
    folds,
    params_idx
):
    def score(seed):
        projector = ProjectorFactory.create(
            proj_name, n_components, seed, **proj_hyperparams
        )
        measurements = [
            MeasurementFactory.create(c.name, seed, **c.params)
            for c in measurement_configs
        ]
        scores = score_projector_config(
            embeddings_future, projector, measurements, folds
        )
        return scores

    def summarize(results):
        keys = results[0].keys()
        summary = {}

        for k in keys:
            values = np.array([r[k] for r in results])
            summary[f"{k}-mean"] = values.mean()
            summary[f"{k}-std"] = values.std(ddof=1)  # sample std

        return summary

    # compute the scores for the measurements over all the seeds
    scores_for_seeds = [score(seed) for seed in seeds]

    # report mean ± std over N runs
    results = summarize(scores_for_seeds)

    # add the meta data
    meta_data = {
        "projector": proj_name,
        "dataset": dataset_name,
        "n_components": n_components,
        "num_seeds": len(seeds),
        "params_idx": params_idx
    }
    return meta_data | proj_hyperparams | results


def make_dask_client():
    cluster = LocalCUDACluster(
        threads_per_worker=1,
    )
    client = Client(cluster)
    return client, cluster


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to a YAML configuration file.",
)
def calibrate(config):
    # set up the paths and the config
    paths = Paths.from_env()
    cfg = Config.from_yaml(config)

    # create a run id
    # __dict__ isn't recursive - TODO: fix
    run_id = generate_run_id("calidration", config)
    
    # set up the logger
    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running the calibration step of the experiment.")
    logger.info(f"Run id is {run_id}.")

    # load in the datasets training split
    train_embeddings = {
        f"{dataset}_{encoder}": paths.embedding(dataset, encoder).split("train").load()
        for dataset, encoder in product(cfg.datasets, cfg.encoders)
    }

    # stratified subsample to the amount specifier in the config if required
    num_subsamples = cfg.calibration.subsample
    if num_subsamples:
        train_embeddings = {
            name: embeddings.stratified_sample(num_subsamples)
            for name, embeddings in train_embeddings.items()
        }

    # generate the folds
    # note that the folds should be generated from the sampled training set, not the full training set
    # they are generated once using a fixed seed, the paired seeds are used to the projectors
    # the fold seed is set to 42 for reproducability over the stages
    num_folds = cfg.num_folds
    dataset_folds = {
        name: generate_stratified_folds(ds, n_splits=num_folds, seed=42)
        for name, ds in train_embeddings.items()
    }

    # set up the parrallel client
    client, cluster = make_dask_client()

    # timings
    start_time = perf_counter()

    # for each projector in the calibration config
    results = {}

    for projector in cfg.calibration.projectors:
        logger.info(
            f"Evaluating projector: {projector.name}: hyperparameters {projector.params}"
        )
        if len(projector.params) == 0:
            logger.info("No hyperparameters to tune, skipping.")
            continue

        # geneate the hyperparameter grid for this projector
        param_grid = make_param_grid(projector.params)
        logger.info(
            f"Generated {len(param_grid)} hyperparameter combinations to evaluate."
        )

        # we want the calibration for each projector over all datasets
        projector_results = []
        for dataset_name, embeddings in train_embeddings.items():
            logger.info(f"Evaluating projector {projector.name} on dataset: {dataset_name}")
            logger.info(f"Number of samples: {embeddings.num_samples()}")


            embeddings_future = client.scatter(embeddings, broadcast=False)
            folds = dataset_folds[dataset_name]
            seeds = cfg.seeds.calibration

            # for the dimensions, we ensure that the full size of the embeddings is present
            dimensions = cfg.calibration.dimensions
            # embedding_size = embeddings.num_dimensions()
            # if embedding_size not in dimensions:
            #     dimensions.append(embedding_size)

            for n_components in dimensions:
                futures = [
                    client.submit(
                        score_projector_config_task,
                        embeddings_future,
                        dataset_name,
                        projector.name,
                        projector_hyperparams,
                        cfg.calibration.measurements,
                        n_components,
                        seeds,
                        folds,
                        params_idx, # params index allows us to analyse based on the mean for params
                        pure=False,
                    )
                    for params_idx, projector_hyperparams in enumerate(param_grid)
                ]

                desc = f"{projector.name} | {dataset_name} | d={n_components}"
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    projector_results.append(future.result())

                    #projector_results.extend(client.gather(futures))

        results["projector"] = pd.DataFrame(projector_results)

    # save the results per projector
    for projector, results_df in results.items():
        output_path = paths.run(run_id).calibration().projector(projector)
        output_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path / 'scores.csv', index=False)

        # find the best hyperparameters for each projector, for each dataset
        # 1. compute the average score for each hyperparameter set (based on the params idx) over the dims
        # 2. select the top performing one
        score_to_select = cfg.calibration.select
        mean_per_hyperparams = results_df.groupby('params_idx')[score_to_select].mean()
        best_params_idx = mean_per_hyperparams.idxmax()
        best_row = results_df[results_df["params_idx"] == best_params_idx].iloc[0]
        best_row_json = best_row.to_dict()

        logger.info(f"The best hyperparameters for the {projector} projector were:")
        logger.info(best_row_json)

        # save
        with open(output_path / "best_params.json", "w") as f:
            json.dump(best_row_json, f, indent=2)


    elapsed_time = perf_counter() - start_time
    logger.info(f"\nTotal time: {elapsed_time:.2f} seconds")
