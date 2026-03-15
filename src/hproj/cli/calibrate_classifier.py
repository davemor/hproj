from dataclasses import dataclass
from itertools import product
from time import perf_counter

import click

from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.dask import make_dask_client
from hproj.util.logging import setup_logging


@dataclass
class TaskDescription:
    projector_name: str
    projector_param: dict
    


@click.command()
@click.option(
    "--run-id", 
    "-r",
    type=str,
    help="The id of a run to continue.",
    required=True,
    default=None
)
def calibrate_classifier(run_id: str):

    # set up the run_id
    paths = Paths.from_env()

    # load the config from the run dir
    cfg = Config.from_yaml(paths.data_root.run(run_id).config())

    # set up the logger
    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running the calibration classifier step of the experiment.")
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

    projector_cfgs = []
    for projector in cfg.calibration.projectors:
        # load the projector configuration



    try:
        # timings
        start_time = perf_counter()

        # for each classifier

            # generate a hyperparameter grid

            # generate the task descriptions

            # for each projector (using the fixed configuration from earlier)

                    # for each dataset

                        # for each dimension

                            # for each seed

                                # for each configuration of hyperparameters

                                        # create a task specification

            # submit all the task specifications

                # in the task, run all the classifiers and compute all the classification metrics

            # put the results in a data frame and save it

            # do the aggrigation to get the best over all hyperparameter configuration

    finally:
        client.close()
        cluster.close()