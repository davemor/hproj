import click
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.logging import setup_logging


ENCODER_DISPLAY = {
    "uni": "UNI",
    "uni2": "UNI2",
    "hibou": "Hibou",
    "dinov2": "DINOv2",
}

ENCODER_ORDER = ["uni", "uni2", "hibou", "dinov2"]

DATASET_DISPLAY = {
    "kather100k": "Kather",
    "spider-colorectal": "CRC",
    "spider-breast": "Breast",
    "spider-skin": "Skin",
    "spider-thorax": "Thorax",
}

DATASET_ORDER = ["kather100k", "spider-colorectal", "spider-breast", "spider-skin", "spider-thorax"]

PROJECTOR_DISPLAY = {
    "umap": "UMAP",
    "pca": "PCA",
    "grp": "GRP",
}

PROJECTOR_ORDER = ["umap", "pca", "grp"]

PROJECTOR_SHORT = {
    "pca": "P",
    "umap": "U",
    "grp": "G",
}

METRIC_DISPLAY = {
    "roc_auc": "ROC-AUC",
    "accuracy": "Acc",
}

ESTIMATOR_COLS = [
    ("intrinsic_dim_mle", "MLE"),
    ("intrinsic_dim_danco", "DANCo"),
    ("intrinsic_dim_twonn", "TwoNN"),
    ("intrinsic_dim_chavez", "Chávez"),
]


def sanitize_label(text: str) -> str:
    """Title-case a label, replacing underscores/hyphens with spaces.

    Preserves words that are already all-uppercase (e.g. 'AUC', 'ROC-AUC').
    """
    words = text.replace("_", " ").replace("-", " ").split()
    return " ".join(w if w.isupper() else w.title() for w in words)


def format_projector_name(name: str) -> str:
    """Capitalize projector initials (e.g. pca -> PCA, umap -> UMAP)."""
    return name.upper()


def detect_metrics(df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    """Detect metric columns from dataset summary CSV.

    Returns list of (mean_col, std_col, base_metric, display_label) tuples.
    """
    mean_suffix = "_seed_mean_dataset_mean"
    std_suffix = "_seed_mean_dataset_std"
    group_cols = {"classifier", "projector", "n_components"}
    metrics = []
    for col in df.columns:
        if col.endswith(mean_suffix) and col not in group_cols:
            base = col[: -len(mean_suffix)]
            std_col = f"{base}{std_suffix}"
            if std_col in df.columns:
                metrics.append((col, std_col, base, METRIC_DISPLAY.get(base, base)))
    return metrics


def _plot_with_fill(
    ax: plt.Axes,
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    projector_order: list[str],
):
    """Plot lines with ±1 std shaded fill for each projector."""
    for proj in projector_order:
        proj_df = df[df["Projector"] == proj].sort_values("n_components")
        if proj_df.empty:
            continue
        x = proj_df["n_components"]
        y = proj_df[mean_col]
        std = proj_df[std_col]
        line = ax.plot(x, y, label=proj)[0]
        ax.fill_between(x, y - std, y + std, alpha=0.2, color=line.get_color())


def plot_metric_side_by_side(
    df: pd.DataFrame,
    mean_col: str,
    std_col: str,
    metric_label: str,
    detail_dims: int,
    output_path: Path,
    fontsize: int = 14,
    ticksize: int = 14,
    figsize: tuple[int, int] = (11, 5),
):
    """Create a side-by-side plot: full range (left) and detail view (right).

    Uses precomputed mean/std columns from the dataset summary for error bands.
    """
    output_path = Path(output_path)

    plot_df = df.copy()
    plot_df["Projector"] = plot_df["projector"].apply(format_projector_name)

    proj_order = [p.upper() for p in PROJECTOR_ORDER if p.upper() in plot_df["Projector"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plt.subplots_adjust(top=0.85, bottom=0.2)

    max_dim = int(plot_df["n_components"].max())
    y_label = sanitize_label(metric_label)

    # Left: full range
    ax_full = axes[0]
    ax_full.set_xlim(1, max_dim)
    ax_full.set_title("Full Dimensions", fontsize=fontsize + 2)
    ax_full.set_xlabel("Dimensions", fontsize=fontsize)
    ax_full.set_ylabel(y_label, fontsize=fontsize)
    ax_full.tick_params(labelsize=ticksize)
    _plot_with_fill(ax_full, plot_df, mean_col, std_col, proj_order)

    # Right: detail view (up to detail_dims)
    ax_detail = axes[1]
    detail_df = plot_df[plot_df["n_components"] <= detail_dims]
    ax_detail.set_xlim(1, detail_dims)
    ax_detail.set_title("Lower Dimensions", fontsize=fontsize + 2)
    ax_detail.set_xlabel("Dimensions", fontsize=fontsize)
    ax_detail.set_ylabel(y_label, fontsize=fontsize)
    ax_detail.tick_params(labelsize=ticksize)
    _plot_with_fill(ax_detail, detail_df, mean_col, std_col, proj_order)

    # Shared legend below
    handles, labels = ax_full.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Projector",
        title_fontsize=fontsize,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, -0.02),
        loc="upper center",
        ncol=len(labels),
    )

    fig.suptitle(
        f"{y_label} vs Dimensions",
        fontsize=fontsize + 2,
        fontweight="bold",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_intrinsic_dims_df(id_df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy DataFrame of intrinsic dimensionality estimates."""
    keep_cols = ["dataset", "encoder"]
    if "num_dimensions" in id_df.columns:
        keep_cols.append("num_dimensions")
    for col, _ in ESTIMATOR_COLS:
        if col in id_df.columns:
            keep_cols.append(col)
    return id_df[keep_cols].copy()


def intrinsic_dims_df_to_tex(df: pd.DataFrame) -> str:
    """Convert an intrinsic dims DataFrame to a LaTeX table string."""
    present = [(col, label) for col, label in ESTIMATOR_COLS if col in df.columns]
    has_D = "num_dimensions" in df.columns
    n_est = len(present)
    col_spec = "l l" + (" r" if has_D else "") + " r" * n_est

    lines = []
    lines.append(r"\begin{table}[h]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header_cells = [r"\textbf{Dataset}", r"\textbf{Encoder}"]
    if has_D:
        header_cells.append(r"\textbf{D}")
    header_cells += [rf"\textbf{{{label}}}" for _, label in present]
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    datasets = df["dataset"].unique()
    for i, dataset in enumerate(datasets):
        subset = df[df["dataset"] == dataset]
        n_rows = len(subset)
        for j, (_, row) in enumerate(subset.iterrows()):
            cells = []
            if j == 0:
                if n_rows > 1:
                    cells.append(rf"\multirow{{{n_rows}}}{{*}}{{{dataset}}}")
                else:
                    cells.append(str(dataset))
            else:
                cells.append("")
            cells.append(str(row["encoder"]))
            if has_D:
                cells.append(str(int(row["num_dimensions"])))
            for col, _ in present:
                cells.append(f"{row[col]:.2f}")
            lines.append(" & ".join(cells) + r" \\")
        if i < len(datasets) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")
    lines.append(r"\caption{Intrinsic dimensionality estimates per dataset and encoder.}")
    lines.append(r"\label{tab:intrinsic_dims}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_dim_summary_df(
    thresh_df: pd.DataFrame,
    id_df: pd.DataFrame | None,
    metric: str,
    condition: float,
    classifier: str,
) -> pd.DataFrame:
    """Build a tidy DataFrame of minimum dimensions for threshold recovery.

    Returns a DataFrame with columns:
        dataset, encoder, d_id, full_d, projector, achieved_dim
    """
    tdf = thresh_df[
        (thresh_df["metric"] == metric)
        & (thresh_df["condition"] == condition)
        & (thresh_df["classifier"] == classifier)
    ].copy()

    all_encoders = set(tdf["encoder"].unique())
    if id_df is not None:
        all_encoders |= set(id_df["encoder"].unique())
    encoders = [e for e in ENCODER_ORDER if e in all_encoders]
    if not encoders:
        encoders = sorted(all_encoders)

    projectors = [p for p in PROJECTOR_ORDER if p in tdf["projector"].unique()]

    all_datasets = set(tdf["dataset"].unique())
    if id_df is not None:
        all_datasets |= set(id_df["dataset"].unique())
    datasets = [d for d in DATASET_ORDER if d in all_datasets]
    for d in sorted(all_datasets):
        if d not in datasets:
            datasets.append(d)

    id_map: dict[tuple[str, str], int] = {}
    full_d_map: dict[str, int] = {}
    if id_df is not None:
        for _, row in id_df.iterrows():
            id_map[(row["dataset"], row["encoder"])] = int(round(row["intrinsic_dim_mle"]))
            if "num_dimensions" in row.index and pd.notna(row.get("num_dimensions")):
                full_d_map[row["encoder"]] = int(row["num_dimensions"])

    thresh_lookup: dict[tuple[str, str, str], float] = {}
    for _, row in tdf.iterrows():
        thresh_lookup[(row["dataset"], row["encoder"], row["projector"])] = row["achieved_dim"]

    rows = []
    for dataset in datasets:
        for enc in encoders:
            d_id = id_map.get((dataset, enc))
            full_d = full_d_map.get(enc)
            for proj in projectors:
                achieved = thresh_lookup.get((dataset, enc, proj))
                rows.append({
                    "dataset": dataset,
                    "encoder": enc,
                    "d_id": int(d_id) if d_id is not None else None,
                    "full_d": int(full_d) if full_d is not None else None,
                    "projector": proj,
                    "achieved_dim": (
                        int(achieved)
                        if achieved is not None and not pd.isna(achieved)
                        else None
                    ),
                })
    return pd.DataFrame(rows)


def dim_summary_df_to_tex(
    df: pd.DataFrame,
    metric: str,
    condition: float,
) -> str:
    """Convert a tidy dim summary DataFrame to a LaTeX table string."""
    metric_label = METRIC_DISPLAY.get(metric, metric)
    pct = int(round(condition * 100))

    encoders = [e for e in ENCODER_ORDER if e in df["encoder"].unique()]
    if not encoders:
        encoders = sorted(df["encoder"].unique())
    projectors = [p for p in PROJECTOR_ORDER if p in df["projector"].unique()]
    if not projectors:
        projectors = sorted(df["projector"].unique())
    datasets = [d for d in DATASET_ORDER if d in df["dataset"].unique()]
    for d in sorted(df["dataset"].unique()):
        if d not in datasets:
            datasets.append(d)

    id_map: dict[tuple[str, str], int] = {}
    full_d_map: dict[str, int] = {}
    thresh_lookup: dict[tuple[str, str, str], int] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("d_id")):
            id_map[(row["dataset"], row["encoder"])] = int(row["d_id"])
        if pd.notna(row.get("full_d")):
            full_d_map[row["encoder"]] = int(row["full_d"])
        if pd.notna(row.get("achieved_dim")):
            thresh_lookup[(row["dataset"], row["encoder"], row["projector"])] = int(
                row["achieved_dim"]
            )

    n_sub = 1 + len(projectors)
    n_encoders = len(encoders)
    col_spec = "l" + "c" * (n_encoders * n_sub)

    def fmt_dim(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "---"
        return str(int(val))

    lines = []
    lines.append(r"\begin{table}[!htpb]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header1_cells = [""]
    for enc in encoders:
        enc_display = ENCODER_DISPLAY.get(enc, enc)
        header1_cells.append(rf"\multicolumn{{{n_sub}}}{{c}}{{{enc_display}}}")
    lines.append(" & ".join(header1_cells) + r" \\")

    cmidrules = []
    for i in range(n_encoders):
        start = 2 + i * n_sub
        end = start + n_sub - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
    lines.append(" ".join(cmidrules))

    header2_cells = ["Dataset"]
    for _ in encoders:
        header2_cells.append(r"$d_\text{ID}$")
        for proj in projectors:
            header2_cells.append(PROJECTOR_SHORT.get(proj, proj[0].upper()))
    lines.append(" & ".join(header2_cells) + r" \\")

    lines.append(r"\midrule")

    for dataset in datasets:
        ds_display = DATASET_DISPLAY.get(dataset, sanitize_label(dataset))
        cells = [ds_display]
        for enc in encoders:
            d_id = id_map.get((dataset, enc))
            cells.append(fmt_dim(d_id))
            for proj in projectors:
                dim = thresh_lookup.get((dataset, enc, proj))
                cells.append(fmt_dim(dim))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")

    dim_groups: dict[int, list[str]] = {}
    for enc, d in full_d_map.items():
        dim_groups.setdefault(d, []).append(ENCODER_DISPLAY.get(enc, enc))
    dim_desc = " or ".join(
        f"{d}-D ({', '.join(names)})" for d, names in sorted(dim_groups.items())
    )
    proj_legend = ". ".join(
        f"{PROJECTOR_SHORT.get(p, p[0].upper())}: {PROJECTOR_DISPLAY.get(p, p.upper())}"
        for p in projectors
    )
    caption = (
        rf"Minimum dimensions for {pct}\% {metric_label} recovery with "
        rf"intrinsic dimensionality included for comparisons, where {pct}\% is "
        rf"measured relative to the full-dimensional baseline. "
        rf"$d_\text{{ID}}$: MLE intrinsic dimensionality. {proj_legend}. "
        rf"``---'': {pct}\% never reached."
    )
    if dim_desc:
        caption += f" Full-dimensional embeddings are {dim_desc}."

    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{tab:dim{pct}-summary}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def make_generalisation_df(
    gen_df: pd.DataFrame,
    thresh_df: pd.DataFrame | None,
    id_df: pd.DataFrame | None,
    dataset: str,
    metric: str,
    classifier: str,
) -> pd.DataFrame:
    """Build a flat DataFrame of generalisation results for one dataset/metric/classifier.

    Returns a DataFrame with columns:
        projector, encoder, metric_full, metric_1d, metric_id, d_id, d_99, d_95, full_d
    """
    id_map: dict[str, int] = {}
    dim_map: dict[str, int] = {}
    if id_df is not None:
        ds_id = id_df[id_df["dataset"] == dataset]
        for _, row in ds_id.iterrows():
            id_map[row["encoder"]] = int(round(row["intrinsic_dim_mle"]))
            if "num_dimensions" in row and pd.notna(row.get("num_dimensions")):
                dim_map[row["encoder"]] = int(row["num_dimensions"])

    gdf = gen_df[
        (gen_df["dataset"] == dataset) & (gen_df["classifier"] == classifier)
    ].copy()

    group_cols = ["dataset", "encoder", "n_components", "projector"]
    gdf_mean = gdf.groupby(group_cols, as_index=False)[metric].mean()

    thresh_lookup: dict[tuple[str, str], dict[float, float]] = {}
    if thresh_df is not None:
        tdf = thresh_df[
            (thresh_df["dataset"] == dataset)
            & (thresh_df["classifier"] == classifier)
            & (thresh_df["metric"] == metric)
        ]
        for _, row in tdf.iterrows():
            key = (row["encoder"], row["projector"])
            thresh_lookup.setdefault(key, {})[row["condition"]] = row["achieved_dim"]

    projectors_present = [p for p in PROJECTOR_ORDER if p in gdf["projector"].unique()]
    encoders_present = gdf["encoder"].unique().tolist()

    def get_metric_val(encoder, projector, n_components):
        rows = gdf_mean[
            (gdf_mean["encoder"] == encoder)
            & (gdf_mean["projector"] == projector)
            & (gdf_mean["n_components"] == n_components)
        ]
        return rows[metric].iloc[0] if not rows.empty else None

    def get_full_d_val(encoder):
        rows = gdf_mean[
            (gdf_mean["encoder"] == encoder) & (gdf_mean["projector"] == "none")
        ]
        return rows[metric].iloc[0] if not rows.empty else None

    if not dim_map:
        for enc in encoders_present:
            none_rows = gdf[(gdf["encoder"] == enc) & (gdf["projector"] == "none")]
            if not none_rows.empty:
                dim_map[enc] = int(none_rows["n_components"].iloc[0])

    result_rows = []
    for projector in projectors_present:
        for encoder in encoders_present:
            d_id = id_map.get(encoder)
            full_d = dim_map.get(encoder)
            val_full = get_full_d_val(encoder)
            val_1d = get_metric_val(encoder, projector, 1)
            val_id = get_metric_val(encoder, projector, d_id) if d_id is not None else None
            thresh_entry = thresh_lookup.get((encoder, projector), {})
            d_99 = thresh_entry.get(0.99)
            d_95 = thresh_entry.get(0.95)
            result_rows.append({
                "projector": projector,
                "encoder": encoder,
                "metric_full": val_full,
                "metric_1d": val_1d,
                "metric_id": val_id,
                "d_id": int(d_id) if d_id is not None else None,
                "d_99": (
                    int(d_99) if d_99 is not None and not pd.isna(d_99) else None
                ),
                "d_95": (
                    int(d_95) if d_95 is not None and not pd.isna(d_95) else None
                ),
                "full_d": int(full_d) if full_d is not None else None,
            })
    return pd.DataFrame(result_rows)


def generalisation_df_to_tex(
    df: pd.DataFrame,
    dataset: str,
    metric: str,
    classifier: str,
) -> str:
    """Convert a generalisation DataFrame to a LaTeX table string."""
    metric_label = METRIC_DISPLAY.get(metric, metric)

    projectors_present = [p for p in PROJECTOR_ORDER if p in df["projector"].unique()]
    encoders_present = df["encoder"].unique().tolist()

    dim_map: dict[str, int] = {}
    for _, row in df.iterrows():
        if pd.notna(row.get("full_d")):
            dim_map[row["encoder"]] = int(row["full_d"])

    def fmt_metric(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "---"
        return f"{val:.3f}"

    def fmt_dim(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "---"
        return str(int(val))

    dim_groups: dict[int, list[str]] = {}
    for enc, d in dim_map.items():
        dim_groups.setdefault(d, []).append(ENCODER_DISPLAY.get(enc, enc))
    dim_desc = " or ".join(
        f"{d}-D ({', '.join(names)})" for d, names in sorted(dim_groups.items())
    )

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(
        r"\begin{tabular}{ll @{\hskip 10pt} c @{\hskip 6pt} c @{\hskip 6pt} c"
        r" @{\hskip 10pt} c @{\hskip 10pt} c @{\hskip 6pt} c}"
    )
    lines.append(r"\toprule")

    lines.append(
        rf"Projector & Encoder"
        rf" & {metric_label} & {metric_label} & {metric_label}"
        rf" & $d_{{\text{{ID}}}}$ & $d_{{99}}$ & $d_{{95}}$ \\"
    )
    lines.append(
        r"          &        "
        rf" & @$d_{{\text{{full}}}}$ & @$d_{{\text{{1d}}}}$ & @$d_{{\text{{ID}}}}$"
        r" &  &  &  \\"
    )
    lines.append(r"\midrule")

    for pi, projector in enumerate(projectors_present):
        n_enc = len(encoders_present)
        proj_display = PROJECTOR_DISPLAY.get(projector, projector.upper())

        for ei, encoder in enumerate(encoders_present):
            enc_display = ENCODER_DISPLAY.get(encoder, encoder)
            row = df[(df["projector"] == projector) & (df["encoder"] == encoder)]
            if row.empty:
                continue
            row = row.iloc[0]

            cells = []
            if ei == 0:
                cells.append(rf"\multirow{{{n_enc}}}{{*}}{{{proj_display}}}")
            else:
                cells.append("")
            cells.append(enc_display)
            cells.append(fmt_metric(row.get("metric_full")))
            cells.append(fmt_metric(row.get("metric_1d")))
            cells.append(fmt_metric(row.get("metric_id")))
            cells.append(fmt_dim(row.get("d_id")))
            cells.append(fmt_dim(row.get("d_99")))
            cells.append(fmt_dim(row.get("d_95")))
            lines.append(" & ".join(cells) + r" \\")

        if pi < len(projectors_present) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\vspace{2pt}")

    dataset_display = DATASET_DISPLAY.get(dataset, sanitize_label(dataset))
    classifier_display = sanitize_label(classifier)
    caption_line = (
        rf"\caption{{{dataset_display} {metric_label} results across projectors and encoders. "
        rf"$d_{{\text{{ID}}}}$: MLE intrinsic dimensionality. "
        rf"$d_{{99}}$, $d_{{95}}$: minimum dimension at which {classifier_display} "
        rf"{metric_label} reaches 99\% or 95\% of {metric_label} @Full-D. "
        rf"``---'': threshold never reached."
    )
    if dim_desc:
        caption_line += f" Full-dimensional embeddings are {dim_desc}."
    caption_line += "}"
    lines.append(caption_line)

    ds_tag = dataset.replace("-", "_")
    lines.append(rf"\label{{tab:{ds_tag}_{metric}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to report on.")
def report(run_id: str):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Generating report fragments.")
    logger.info(f"Run id is {run_id}.")

    report_dir = run_paths.root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    all_tables = []  # collect all generated tables for combined output

    # intrinsic dimensionality table
    id_path = run_paths.intrinsic_dims()
    if id_path.exists():
        id_df = pd.read_csv(id_path)
        id_table_df = make_intrinsic_dims_df(id_df)
        csv_out = report_dir / "intrinsic_dims_table.csv"
        id_table_df.to_csv(csv_out, index=False)
        logger.info(f"Wrote intrinsic dims DataFrame to {csv_out}")
        table_tex = intrinsic_dims_df_to_tex(id_table_df)
        out = report_dir / "intrinsic_dims_table.tex"
        out.write_text(table_tex)
        all_tables.append(f"% === intrinsic_dims_table ===\n{table_tex}")
        logger.info(f"Wrote intrinsic dims table to {out}")
    else:
        logger.warning("No intrinsic_dims.csv found, skipping intrinsic dims table.")

    # side-by-side dimension plots per classifier and metric
    detail_dims = cfg.report.detail_dims
    report_plots_dir = report_dir / "plots"
    report_plots_dir.mkdir(parents=True, exist_ok=True)

    summaries_dir = run_paths.curve().summaries().root
    if summaries_dir.exists():
        for classifier in cfg.curve.classifiers:
            summary_path = run_paths.curve().summaries().classifier_dataset_summary(classifier)
            if not summary_path.exists():
                logger.warning(f"No dataset summary found for {classifier}, skipping plots.")
                continue

            summary_df = pd.read_csv(summary_path)
            metrics = detect_metrics(summary_df)
            logger.info(
                f"Generating {len(metrics)} plot(s) for classifier '{classifier}': "
                f"{[label for _, _, _, label in metrics]}"
            )

            for mean_col, std_col, base_metric, metric_label in metrics:
                out_path = report_plots_dir / f"{classifier}_{base_metric}.pdf"
                plot_metric_side_by_side(
                    summary_df,
                    mean_col=mean_col,
                    std_col=std_col,
                    metric_label=metric_label,
                    detail_dims=detail_dims,
                    output_path=out_path,
                )
                logger.info(f"Wrote plot to {out_path}")
    else:
        logger.warning("No curve summaries directory found, skipping dimension plots.")

    # dim summary tables (cross-tabulated by encoder)
    thresh_path_summary = run_paths.curve().root / "thresholds_results.csv"
    id_path_summary = run_paths.intrinsic_dims()
    if thresh_path_summary.exists():
        thresh_df_summary = pd.read_csv(thresh_path_summary)
        id_df_summary = pd.read_csv(id_path_summary) if id_path_summary.exists() else None

        for condition in cfg.thresholds.conditions:
            for metric in cfg.thresholds.metrics:
                for classifier in cfg.curve.classifiers:
                    pct = int(round(condition * 100))
                    tag = f"dim{pct}_summary_{metric}_{classifier}"
                    summary_df = make_dim_summary_df(
                        thresh_df_summary,
                        id_df_summary,
                        metric=metric,
                        condition=condition,
                        classifier=classifier,
                    )
                    csv_out = report_dir / f"{tag}.csv"
                    summary_df.to_csv(csv_out, index=False)
                    logger.info(f"Wrote dim summary DataFrame to {csv_out}")
                    table_tex = dim_summary_df_to_tex(summary_df, metric=metric, condition=condition)
                    out = report_dir / f"{tag}.tex"
                    out.write_text(table_tex)
                    all_tables.append(f"% === {tag} ===\n{table_tex}")
                    logger.info(f"Wrote dim summary table to {out}")
    else:
        logger.warning("No thresholds_results.csv found, skipping dim summary tables.")

    # generalisation tables per dataset, metric, and classifier
    gen_path = run_paths.root / "generalisation_results.csv"
    thresh_path = run_paths.curve().root / "thresholds_results.csv"
    id_path = run_paths.intrinsic_dims()

    if gen_path.exists():
        gen_df = pd.read_csv(gen_path)
        thresh_df = pd.read_csv(thresh_path) if thresh_path.exists() else None
        id_df = pd.read_csv(id_path) if id_path.exists() else None

        # determine metrics from generalisation columns (skip non-metric cols)
        skip_cols = {
            "dataset", "encoder", "n_components", "projector", "classifier",
            "best_params", "cv_score", "base_seed", "confusion_matrix",
        }
        gen_metrics = [c for c in gen_df.columns if c not in skip_cols]

        # determine classifiers
        classifiers = gen_df["classifier"].unique()

        for classifier in classifiers:
            for metric in gen_metrics:
                for dataset in cfg.datasets:
                    tag = f"{dataset}_{metric}_{classifier}"
                    gen_table_df = make_generalisation_df(
                        gen_df, thresh_df, id_df,
                        dataset=dataset,
                        metric=metric,
                        classifier=classifier,
                    )
                    csv_out = report_dir / f"generalisation_{tag}.csv"
                    gen_table_df.to_csv(csv_out, index=False)
                    logger.info(f"Wrote generalisation DataFrame to {csv_out}")
                    table_tex = generalisation_df_to_tex(
                        gen_table_df,
                        dataset=dataset,
                        metric=metric,
                        classifier=classifier,
                    )
                    out = report_dir / f"generalisation_{tag}.tex"
                    out.write_text(table_tex)
                    all_tables.append(f"% === generalisation_{tag} ===\n{table_tex}")
                    logger.info(f"Wrote generalisation table to {out}")
    else:
        logger.warning("No generalisation_results.csv found, skipping generalisation tables.")

    # write combined tables file for debugging
    if all_tables:
        all_tables_path = report_dir / "all_tables.tex"
        all_tables_path.write_text("\n\n".join(all_tables) + "\n")
        logger.info(f"Wrote {len(all_tables)} tables to {all_tables_path}")

    logger.info("Report generation complete.")