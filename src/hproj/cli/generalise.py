from itertools import product
from pathlib import Path
from time import perf_counter

import click
import cupy as cp
import pandas as pd
from dask.distributed import as_completed
from tqdm.auto import tqdm

from hproj.classifiers.classifier import ClassifierFactory
from hproj.classifiers.metrics import compute_classification_metrics
from hproj.data.feature_space import FeatureSpace
from hproj.data.folds import generate_stratified_folds
from hproj.data.paths import Paths
from hproj.projectors.projector import ProjectorFactory
from hproj.util.config import Config
from hproj.util.dask import make_dask_client
from hproj.util.hyperparams import make_param_grid
from hproj.util.logging import setup_logging
from hproj.util.seeds import make_seed
from hproj.util.timing import append_stage_time


def evaluate_generalisation_task(
    train_embeddings,
    test_embeddings,
    folds,
    dataset_name: str,
    encoder: str,
    n_components: int,
    projector_name: str,
    projector_params: dict,
    classifier_name: str,
    classifier_param_grid: list[dict],
    base_seed: int,
    select_metric: str,
    use_projector: bool = True,
):
    """
    Evaluate a projector/classifier combination at a single dimension.

    1. Project (or skip for D) train and test data.
    2. Tune classifier hyperparameters via cross-validation on training folds.
    3. Retrain on full projected training set with best params.
    4. Evaluate on projected test set.
    """
    from sklearn.metrics import confusion_matrix

    seed = make_seed(
        base_seed,
        dataset_name,
        encoder,
        n_components,
        projector_name,
        classifier_name,
    )

    # step 1: project data (or use raw embeddings for full dimension D)
    if use_projector:
        projector = ProjectorFactory.create(
            projector_name, n_components, seed, **projector_params
        )
        projector.fit(train_embeddings)
        train_data = projector.transform(train_embeddings)
        test_data = projector.transform(test_embeddings)
    else:
        train_data = train_embeddings
        test_data = test_embeddings

    # step 2: tune classifier hyperparameters via cross-validation
    best_score = -float("inf")
    best_params = classifier_param_grid[0] if classifier_param_grid else {}

    for params in classifier_param_grid:
        fold_scores = []
        for train_idx, val_idx in folds:
            train_fold = FeatureSpace(train_data.features[train_idx], train_data.labels[train_idx])
            val_fold = FeatureSpace(train_data.features[val_idx], train_data.labels[val_idx])

            clf = ClassifierFactory.create(classifier_name, seed, **params)
            clf.fit(train_fold)
            y_pred, y_scores = clf.predict_and_score(val_fold)
            metrics = compute_classification_metrics(val_fold.labels, y_pred, y_scores)
            fold_scores.append(metrics.get(select_metric, 0.0))

        mean_score = sum(fold_scores) / len(fold_scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # step 3: retrain on full projected training set with best params
    final_clf = ClassifierFactory.create(classifier_name, seed, **best_params)
    final_clf.fit(train_data)

    # step 4: evaluate on projected test set
    y_pred, y_scores = final_clf.predict_and_score(test_data)
    metrics = compute_classification_metrics(test_data.labels, y_pred, y_scores)
    cm = confusion_matrix(
        cp.asnumpy(test_data.labels), cp.asnumpy(y_pred)
    ).tolist()

    return {
        "dataset": dataset_name,
        "encoder": encoder,
        "n_components": n_components,
        "projector": projector_name if use_projector else "none",
        "classifier": classifier_name,
        "best_params": str(best_params),
        "cv_score": best_score,
        "base_seed": base_seed,
        **metrics,
        "confusion_matrix": cm,
    }


@click.command()
@click.option(
    "--run-id", "-r", type=str, required=True, help="The id of a run to evaluate.",
)
@click.option("--force", is_flag=True, help="Recompute existing generalisation outputs.")
def generalise(run_id: str, force: bool):
    paths = Paths.from_env()
    cfg = Config.from_yaml(paths.run(run_id).config())

    logger = setup_logging(paths.run(run_id).log_file())
    logger.info("Running the generalisation evaluation step.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    output_path = paths.run(run_id).root / "generalisation_results.csv"
    if output_path.exists() and not force:
        logger.info(f"Generalisation output already exists at {output_path}, use --force to overwrite.")
        return

    # ---- load embeddings ----
    train_embeddings = {
        f"{ds}_{enc}": paths.embedding(ds, enc).split("train").load()
        for ds, enc in product(cfg.datasets, cfg.encoders)
    }
    test_embeddings = {
        f"{ds}_{enc}": paths.embedding(ds, enc).split("test").load()
        for ds, enc in product(cfg.datasets, cfg.encoders)
    }

    # ---- generate cv folds for each dataset ----
    folds_map = {
        key: generate_stratified_folds(emb, cfg.num_folds, seed=0)
        for key, emb in train_embeddings.items()
    }

    # ---- load intrinsic dimensionality (d_ID) from MLE estimates ----
    id_path = paths.run(run_id).intrinsic_dims()
    intrinsic_dims = {}
    if id_path.exists():
        id_df = pd.read_csv(id_path)
        intrinsic_dims = {
            (row["dataset"], row["encoder"]): int(row["intrinsic_dim_mle"])
            for _, row in id_df.iterrows()
        }

    # ---- load thresholds (d_95, d_99) from curve results ----
    thresh_path = paths.run(run_id).curve().root / "thresholds_results.csv"
    thresh_dims = {}
    if thresh_path.exists():
        thresh_df = pd.read_csv(thresh_path)
        for _, row in thresh_df.iterrows():
            k = (
                row["dataset"],
                row["encoder"],
                row["projector"],
                row["classifier"],
                row["metric"],
                row["condition"],
            )
            thresh_dims[k] = row["achieved_dim"]

    # ---- determine select metric for classifier cv ----
    select_metric = cfg.calibration.classifier.select or "accuracy"

    # ---- build classifier param grids from generalisation config ----
    classifier_grids = {}
    for clf_cfg in cfg.generalisation.classifiers:
        classifier_grids[clf_cfg.name] = make_param_grid(clf_cfg.params)

    # ---- build projector param grids from generalisation config ----
    projector_grids = {}
    for proj_cfg in cfg.generalisation.projectors:
        projector_grids[proj_cfg.name] = make_param_grid(proj_cfg.params)

    # ---- set up dask ----
    client, cluster = make_dask_client()

    try:
        start_time = perf_counter()
        futures = []

        for ds_enc, train_emb in train_embeddings.items():
            dataset, encoder = ds_enc.split("_", 1)
            test_emb = test_embeddings[ds_enc]
            folds = folds_map[ds_enc]

            D = train_emb.num_dimensions()
            d_id = intrinsic_dims.get((dataset, encoder), D)

            for proj_cfg in cfg.generalisation.projectors:
                proj_name = proj_cfg.name
                proj_param_list = projector_grids[proj_name]

                for clf_cfg in cfg.generalisation.classifiers:
                    clf_name = clf_cfg.name
                    clf_param_grid = classifier_grids[clf_name]

                    # look up d_95 and d_99 for this projector/classifier combo
                    # use the first threshold metric configured
                    thresh_metric = (
                        cfg.thresholds.metrics[0]
                        if cfg.thresholds.metrics
                        else select_metric
                    )
                    d_95 = thresh_dims.get(
                        (dataset, encoder, proj_name, clf_name, thresh_metric, 0.95),
                        None,
                    )
                    d_99 = thresh_dims.get(
                        (dataset, encoder, proj_name, clf_name, thresh_metric, 0.99),
                        None,
                    )

                    # build dimension set: {1, d_ID, d_95, d_99, D}
                    dims = {1, d_id, D}
                    if d_95 is not None and not pd.isna(d_95):
                        dims.add(int(d_95))
                    if d_99 is not None and not pd.isna(d_99):
                        dims.add(int(d_99))

                    for n_components in sorted(dims):
                        use_projector = n_components != D

                        # for projected dims, iterate over projector param combos;
                        # for D (no projection), use a single empty-params entry
                        proj_params_iter = proj_param_list if use_projector else [{}]

                        for proj_params in proj_params_iter:
                            for base_seed in cfg.seeds.evaluation:
                                train_f = client.scatter(train_emb, broadcast=True)
                                test_f = client.scatter(test_emb, broadcast=True)
                                folds_f = client.scatter(folds, broadcast=True)

                                future = client.submit(
                                    evaluate_generalisation_task,
                                    train_f,
                                    test_f,
                                    folds_f,
                                    dataset,
                                    encoder,
                                    n_components,
                                    proj_name,
                                    proj_params,
                                    clf_name,
                                    clf_param_grid,
                                    base_seed,
                                    select_metric,
                                    use_projector,
                                    pure=False,
                                )
                                futures.append(future)

        # ---- collect results ----
        all_results = []
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Evaluating generalisation",
        ):
            all_results.append(future.result())

        # ---- save ----
        results_df = pd.DataFrame(all_results)
        output_path = paths.run(run_id).root / "generalisation_results.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Generalisation results saved to {output_path}")

        elapsed = perf_counter() - start_time
        logger.info(f"Total time: {elapsed:.2f} seconds")
        append_stage_time(paths.run(run_id).root, "generalisation", elapsed)

    finally:
        client.close()
        cluster.close()
