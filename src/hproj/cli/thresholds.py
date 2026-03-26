import click
import pandas as pd

from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.logging import setup_logging


def compute_thresholds(
    df_per_dataset: pd.DataFrame,
    unprojected_scores_df: pd.DataFrame,
    thresholds_cfg,
    projector: str,
    classifier: str,
    dataset: str,
    encoder: str,
    dataset_name: str,
):
    results = []
    for metric, condition in zip(thresholds_cfg.metrics, thresholds_cfg.conditions):
        # Find baseline from unprojected
        baseline_row = unprojected_scores_df[
            (unprojected_scores_df['dataset'] == dataset_name) &
            (unprojected_scores_df['classifier'] == classifier)
        ]
        if baseline_row.empty or metric not in baseline_row.columns:
            continue
        baseline = baseline_row[metric].mean()  # average over seeds
        threshold_value = condition

        # Sort by n_components
        sorted_df = df_per_dataset.sort_values('n_components')
        metric_col = f'{metric}_seed_mean'
        if metric_col not in sorted_df.columns:
            continue
        achieved_row = sorted_df[sorted_df[metric_col] >= threshold_value].head(1)
        if achieved_row.empty:
            achieved_dim = float('nan')
            achieved_value = float('nan')
        else:
            achieved_dim = achieved_row['n_components'].iloc[0]
            achieved_value = achieved_row[metric_col].iloc[0]

        results.append({
            'projector': projector,
            'encoder': encoder,
            'classifier': classifier,
            'dataset': dataset,
            'metric': metric,
            'condition': condition,
            'baseline': baseline,
            'threshold_value': threshold_value,
            'achieved_dim': achieved_dim,
            'achieved_value': achieved_value,
        })
    return results


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
@click.option("--force", is_flag=True, help="Recompute existing threshold outputs.")
def thresholds(run_id: str, force: bool):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    curve_paths = run_paths.curve()
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Running the threshold identification phase of the experiment.")
    logger.info(f"Run id is {run_id}.")
    logger.info(f"Force recomputation is {'enabled' if force else 'disabled'}.")

    thresholds_path = curve_paths.root / 'thresholds_results.csv'
    if thresholds_path.exists() and not force:
        logger.info(f"Thresholds output already exists at {thresholds_path}, use --force to overwrite.")
        return

    if not cfg.thresholds.metrics:
        logger.info("No thresholds configured, skipping.")
        return

    all_thresholds = []

    for projector in cfg.curve.projectors:
        for classifier in cfg.curve.classifiers:
            output_path = curve_paths.projector_classifier(projector, classifier)
            seed_summary_path = output_path.seed_summary_results()
            unprojected_path = output_path.unprojected_scores()

            if not seed_summary_path.exists() or not unprojected_path.exists():
                logger.warning(f"Missing results for {projector}/{classifier}, skipping.")
                continue

            seed_summary_df = pd.read_csv(seed_summary_path)
            unprojected_scores_df = pd.read_csv(unprojected_path)

            datasets = seed_summary_df['dataset'].unique()
            for dataset_name in datasets:
                dataset, encoder = dataset_name.split('_', 1)
                df_per_dataset = seed_summary_df[seed_summary_df['dataset'] == dataset_name]
                thresholds = compute_thresholds(df_per_dataset, unprojected_scores_df, cfg.thresholds, projector, classifier, dataset, encoder, dataset_name)
                all_thresholds.extend(thresholds)

    thresholds_df = pd.DataFrame(all_thresholds)
    thresholds_path = curve_paths.root / 'thresholds_results.csv'
    thresholds_df.to_csv(thresholds_path, index=False)
    logger.info(f"Writing thresholds results to {thresholds_path}")

