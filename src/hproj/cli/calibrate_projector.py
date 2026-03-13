import json
import shutil
from itertools import product
from pathlib import Path
from statistics import mean
from time import perf_counter

import click
import pandas as pd
from dask.distributed import Client, as_completed
from dask_cuda import LocalCUDACluster
from tqdm.auto import tqdm

from hproj.data.feature_space import FeatureSpace
from hproj.data.folds import Fold, generate_stratified_folds
from hproj.data.paths import Paths
from hproj.measure.measurement import Measurement, MeasurementFactory
from hproj.projectors.projector import Projector, ProjectorFactory
from hproj.util.atomic import atomic_write_json
from hproj.util.config import Config, MeasurementConfig
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging
from hproj.util.runs import generate_run_id
from hproj.util.seeds import make_seed


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
    base_seed,
    folds,
    params_idx
):
    seed = make_seed(base_seed, dataset_name, proj_name, proj_hyperparams, n_components) 

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

    # add the meta data
    meta_data = {
        "projector": proj_name,
        "dataset": dataset_name,
        "n_components": n_components,
        "base_seed": base_seed,
        "seed": seed,
        "params_idx": params_idx
    }
    return meta_data | proj_hyperparams | scores

def summarize_over_seeds(seed_results_df: pd.DataFrame, measurements: list[Measurement]) -> pd.DataFrame:
    measurement_names = [m.name for m in measurements]
    seed_results_df = seed_results_df.drop('seed', axis=1)  # we want to remote the actualy seed which is not a unique seed id and instead use base_seed
    
    # note that base seed is not in this list because we want to aggrigate over all the seeds
    group_cols = ["projector", "dataset", "n_components", "params_idx"]

    # detect hyperparameter columns automatically
    # so anything that isn't in the:
    # - columns we are going to group by
    # - the base seed (which varies with the measurement in the group)
    # - the column that contains the measurement value (e.g. mean-knn-score)
    hyperparam_cols = [
        c for c in seed_results_df.columns
        if c not in group_cols + ["base_seed"] + measurement_names
    ]

    group_cols = group_cols + hyperparam_cols

    agg = {}
    for col in measurement_names:
        agg[col] = ["mean", "std"]

    summary = seed_results_df.groupby(group_cols).agg(agg)
    
    # flatten multiindex columns
    summary.columns = [
        f"{metric}-{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    # replace NaN std when only one seed
    for col in measurement_names:
        std_col = f"{col}-std"
        summary[std_col] = summary[std_col].fillna(0.0)

    return summary

def make_dask_client():
    cluster = LocalCUDACluster(
        threads_per_worker=1,
    )
    client = Client(cluster)
    return client, cluster


def make_task_key(projector: str, dataset: str, n_components: int, 
                  params_idx: int, seed: int):
    return (
        f"{projector}_{dataset}_d{n_components}"
        f"_p{params_idx}_s{seed}"
    )

def load_task_results(tasks_dir):
    def load_task_result(task_path):
        with open(task_path) as f:
            task_dict = json.load(f)
        return task_dict
    
    task_paths = tasks_dir.glob_tasks()
    results = [load_task_result(p) for p in task_paths]
    results_df =  pd.DataFrame(results)
    print(results_df.head())
    return results_df

def summarize_over_dimensions(projector_results_df, measurements):
    """
    This returns the mean score over all the dimensions for each projector configuration for the each dataset.
    """
    measurement_names = [m.name for m in measurements]

    group_cols = ["projector", "dataset", "params_idx"]
    measurement_cols = [x for name in measurement_names for x in (f"{name}-mean", f"{name}-std")]
    hyperparam_cols = [
        c for c in projector_results_df.columns
        if c not in group_cols + ["n_components"] + measurement_cols
    ]
    group_cols = group_cols + hyperparam_cols

    agg = {}
    for col in measurement_cols:
        agg[col] = ["mean", "std"]

    summary = projector_results_df.groupby(group_cols).agg(agg)
    
    # flatten multiindex columns
    summary.columns = [
        f"{metric}-{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    # TODO: recall that there might be nas in the columns if there is only one dim

    return summary 

def summmrize_over_datasets(projector_results_df, measurements):
    measurement_names = [m.name for m in measurements]

    group_cols = ["projector", "params_idx"]
    measurement_cols = [x for name in measurement_names for x in (f"{name}-mean-mean", f"{name}-mean-std, {name}-std-mean", f"{name}-std-std")]
    hyperparam_cols = [
        c for c in projector_results_df.columns
        if c not in group_cols + ["dataset"] + measurement_cols
    ]
    group_cols = group_cols + hyperparam_cols

    agg = {}
    for col in measurement_cols:
        agg[col] = ["mean", "std"]

    summary = projector_results_df.groupby(group_cols).agg(agg)
    
    # flatten multiindex columns
    summary.columns = [
        f"{metric}-{stat}" for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    # TODO: recall that there might be nas in the columns if there is only one dim

    return summary 


            

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to a YAML configuration file.",
    default=None
)
@click.option(
    "--run-id", 
    "-r",
    type=str,
    help="The id of a run to continue.",
    default=None
)
def calibrate_projector(config: Path, run_id: str):

    # set up the run_id
    assert not(config and run_id), "Use --config for a fresh run and --run-id "
    paths = Paths.from_env()
    if run_id:
        cfg = Config.from_yaml(paths.data_root.run(run_id).config())  # load the config from the run dir
    else:
        run_id = generate_run_id('hproj', config)  # create a new run id
        run_path = paths.run(run_id)
        print(config)
        print(config, run_path.config())
        run_path.mkdir()  # make sure the output dir exists
        shutil.copy(config, run_path.config())  # copy the config into the new runs output directory
        cfg = Config.from_yaml(config)  # load the config from the config path given

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

    try:
        # timings
        start_time = perf_counter()

        for projector in cfg.calibration.projectors:
            logger.info(f"Evaluating projector: {projector.name}")
            logger.info(f"Hyperparams: {projector.params}")
            if len(projector.params) == 0:
                logger.info("No hyperparameters to tune, skipping.")
                continue

            # setup the output path for the projectors calibration data
            output_path = paths.run(run_id).calibration().projector(projector.name)
            output_path.mkdir()
            tasks_dir = output_path.tasks()
            tasks_dir.mkdir()

            # geneate the hyperparameter grid for this projector
            param_grid = make_param_grid(projector.params)
            logger.info(f"Generated {len(param_grid)} hyperparameter configs.")

            # we want the calibration for each projector over all datasets
            seed_level_results = []
            for dataset_name, embeddings in train_embeddings.items():
                logger.info(f"Evaluating projector {projector.name} on dataset: {dataset_name}")
                logger.info(f"Number of samples: {embeddings.num_samples()}")

                embeddings_future = client.scatter(embeddings, broadcast=True)
                folds = dataset_folds[dataset_name]
                base_seeds = cfg.seeds.calibration

                for n_components in cfg.calibration.dimensions:
                    pending_tasks = {}  # future: task_file
                    for params_idx, projector_hyperparams in enumerate(param_grid):
                        for base_seed in base_seeds:
      
                            task_key = make_task_key(projector.name, dataset_name, n_components, params_idx, base_seed)
                            task_file = tasks_dir.task(task_key)
                            if task_file.exists():
                                continue

                            future = client.submit(
                                    score_projector_config_task,
                                    embeddings_future,
                                    dataset_name,
                                    projector.name,
                                    projector_hyperparams,
                                    cfg.calibration.measurements,
                                    n_components,
                                    base_seed,
                                    folds,
                                    params_idx, # params index allows us to analyse based on the mean for params
                                    pure=False)
                        
                            pending_tasks[future] = task_file

                    desc = f"{projector.name} | {dataset_name} | d={n_components}"
                    for future in tqdm(as_completed(pending_tasks.keys()), total=len(pending_tasks), desc=desc):
                        results = future.result()
                        task_file = pending_tasks[future]
                        atomic_write_json(task_file, results)

            # summarise the results for the projector over all the seeds
            results_df = load_task_results(output_path.tasks())

            print(results_df)

            summary_results_df = summarize_over_seeds(results_df, cfg.calibration.measurements)

            print(summary_results_df.head())

            summary_results_df = summarize_over_dimensions(summary_results_df, cfg.calibration.measurements)

            print(summary_results_df.head())

            summary_results_df = summmrize_over_datasets(summary_results_df, cfg.calibration.measurements)  # this is the mean performance for each config over each projector

            print(summary_results_df.head())

            # save over the results
            output_path = paths.run(run_id).calibration().projector(projector.name)
            output_path.mkdir()
            logger.info(f'Saving results to {output_path.root}')
            results_df.to_csv(output_path.seed_scores(), index=False)
            summary_results_df.to_csv(output_path.scores(), index=False)

            # now we have the mean performance of each projector hyperparameter config


            # # find the best hyper parameters and save them to json
            # #   compute the mean for each hyperparameter over each dimension and dataset
            # #       note that we are not doing a per dataset fit
            # #   find the best one
            # #   save to json
            # mean_per_hyperparams = projector_results_df.groupby("params_idx")[cfg.calibration.select].mean()
            # best_params_idx = mean_per_hyperparams.idxmax()
            # best_row = projector_results_df[projector_results_df["params_idx"] == best_params_idx].iloc[0]
            # best_row_json = best_row.to_dict()
            # with open(output_path / "best_params.json", "w") as f:
            #     json.dump(best_row_json, f, indent=2)

            # logger.info(f"The best hyperparameters for the {projector.name} projector were:")
            # logger.info(best_row_json)

        elapsed_time = perf_counter() - start_time
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        logger.info(f"Run output saved to: {run_path.root}")

    finally:
        client.close()
        cluster.close()