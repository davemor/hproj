import cupy as cp
from cuml.metrics import accuracy_score
from sklearn.metrics import roc_auc_score as sk_roc_auc_score


def compute_classification_metrics(
    y_true: cp.ndarray,
    y_pred: cp.ndarray,
    y_scores: cp.ndarray,
) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(
            sk_roc_auc_score(
                cp.asnumpy(y_true),
                cp.asnumpy(y_scores),
                multi_class="ovr",
                average="macro",
            )
        ),
    }