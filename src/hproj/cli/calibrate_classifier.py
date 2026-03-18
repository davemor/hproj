from asyncio import as_completed
from dataclasses import dataclass
from itertools import product
from time import perf_counter

import click
import tqdm

from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.util.atomic import atomic_write_json
from hproj.util.config import Config
from hproj.util.dask import make_dask_client
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging


@dataclass
class TaskDescription:
    classifier_name: str
    classifier_params: str
    dataset: str
    n_components: int
    seed: int
    projector_name: str
    projector_param: dict
    params_idx: int

    def key(self):
        return (
            f"{self.classifer_name}_{self.dataset}_c{self.n_components}"
            f"_hp{self.params_idx}_s{self.seed}_p{self.projector_name}"
        )


def score_classifier_task(embeddings_future, task_desc: TaskDescription, folds):
    # in the task, run all the classifiers and compute all the classification metrics

    


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
        # load the optimal projector config from the projector calibration
        best_params = paths.run(run_id).calibration().projector(projector.name).best_params()
        projector_cfgs.append((projector.name, best_params))


    try:
        # timings
        start_time = perf_counter()

        # for each classifier
        for classifier in cfg.calibration.classifiers:
            logger.info(f"Evaluating classifier: {classifier.name}")
            logger.info(f"Hyperparams: {classifier.params}")
            if len(classifier.params) == 0:
                logger.info("No hyperparameters to tune, skipping.")
                continue

            # setup the output path for the projectors calibration data
            output_path = paths.run(run_id).calibration().classifier(classifier.name)
            output_path.mkdir()
            tasks_dir = output_path.tasks()
            tasks_dir.mkdir()

            # geneate the hyperparameter grid for this projector
            param_grid = make_param_grid(classifier.params)
            logger.info(f"Generated {len(param_grid)} hyperparameter configs.")

            # for each projector (using the fixed configuration from earlier)
            for projector_name, projector_config in projector_cfgs:

                # for each dataset
                for dataset_name, embeddings in train_embeddings.items():
                    logger.info(f"Evaluating projector {projector.name} on dataset: {dataset_name}")
                    logger.info(f"Number of samples: {embeddings.num_samples()}")    

                    embeddings_future = client.scatter(embeddings, broadcast=True)
                    folds = dataset_folds[dataset_name]
                    base_seeds = cfg.seeds.calibration

                    # for each dimension
                    for n_components in cfg.calibration.dimensions:
                        pending_tasks = {}  # future: task_file
                        for params_idx, classifier_params in enumerate(param_grid):
                            for base_seed in base_seeds:
                                # create a task description
                                task_desc = TaskDescription(classifier.name, classifier.params, dataset_name, 
                                                            n_components, base_seed, projector_name, projector_config, 
                                                            params_idx)

                                task_file = tasks_dir.task(task_desc.key())
                                if task_file.exists():
                                    continue

                                future = client.submit(
                                    score_classifier_config_task,
                                    embeddings_future,
                                    task_desc,
                                    folds,
                                    pure=False)
                                
                                pending_tasks[future] = task_file
                        

                        desc = f"{classifier.name} | {dataset_name} | d={n_components}"
                        for future in tqdm(as_completed(pending_tasks.keys()), total=len(pending_tasks), desc=desc):
                            results = future.result()
                            task_file = pending_tasks[future]
                            atomic_write_json(task_file, results)

            # put the results in a data frame and save it

            # do the aggrigation to get the best over all hyperparameter configuration

    finally:
        client.close()
        cluster.close()