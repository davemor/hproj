import click
import numpy as np
import pandas as pd
from skdim.id import MLE
from scipy.spatial.distance import pdist


from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.logging import setup_logging



def chavez_id(X):
    D = pdist(X)
    mu = np.mean(D)
    var = np.var(D)
    return mu**2 / (2 * var)


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
@click.option("--force", is_flag=True, help="Overwrite existing intrinsic dimensions output.")
@click.option("--subsample", type=int, default=None, help="If set, randomly sample this many points before estimation.")
def intrinsic_dims(run_id: str, force: bool, subsample: int):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Running intrinsic dimensionality estimation for run.")
    logger.info(f"Run id is {run_id}.")

    output_path = run_paths.intrinsic_dims()
    if output_path.exists() and not force:
        logger.info(f"Intrinsic dimensions output already exists at {output_path}, use --force to overwrite.")
        return

    rows = []

    for dataset in cfg.datasets:
        for encoder in cfg.encoders:
            embedding_path = paths.embedding(dataset, encoder).split("train")
            logger.info(f"Loading embeddings for {dataset}/{encoder}: {embedding_path.root}")
            feature_space = embedding_path.load()
            features = feature_space.features

            if subsample is not None and len(features) > subsample:
                indices = np.random.default_rng(42).choice(len(features), size=subsample, replace=False)
                features = features[indices]

            # skdim MLE expects numpy arrays
            if hasattr(features, "get"):
                features_np = features.get()  # cupy array
            else:
                features_np = np.asarray(features)

            estimator = MLE()
            estimator.fit(features_np)
            estimated_dim = float(estimator.dimension_)

            estimated_chavez = float(chavez_id(features_np))

            rows.append(
                {
                    "dataset": dataset,
                    "encoder": encoder,
                    "n_samples": len(features_np),
                    "intrinsic_dim_mle": estimated_dim,
                    "intrinsic_dim_chavez": estimated_chavez,
                }
            )

            logger.info(
                f"{dataset}/{encoder}: intrinsic_dim_mle={estimated_dim:.4f}, "
                f"intrinsic_dim_chavez={estimated_chavez:.4f}, n_samples={len(features_np)}"
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)
    logger.info(f"Intrinsic dimensionality results written to {output_path}")
