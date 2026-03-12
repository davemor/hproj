from typing import TYPE_CHECKING

import cupy as cp
from sklearn.model_selection import StratifiedKFold

if TYPE_CHECKING:
    from hproj.data.feature_space import FeatureSpace

Fold = tuple[cp.ndarray, cp.ndarray]


# each dataset has k folds (where k is defined in the config yaml)
# each fold is a tuple of (train_indexs, valid_indexs)
# to get the folds out of the feature space, we can do:
# train_space, valid_space = feature_space.folds(train_indexs, valid_indexs)
def generate_stratified_folds(
    feature_space: "FeatureSpace", n_splits: int, seed: int = 42
) -> list[Fold]:
    # as we are using stratified folds, we need to convert the labels to numpy arrays (as sklearn does not support cupy arrays)
    X = cp.asnumpy(feature_space.features)
    y = cp.asnumpy(feature_space.labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = [(cp.asarray(train), cp.asarray(valid)) for train, valid in skf.split(X, y)]
    return folds
