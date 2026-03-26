import cupy as cp
from cuml.preprocessing import StandardScaler
from xgboost import XGBClassifier

from hproj.classifiers.classifier import Classifier, ClassifierFactory
from hproj.data.feature_space import FeatureSpace


@ClassifierFactory.register('xgboost')
class XGBoostClassifier(Classifier):
    def __init__(self, seed,
                 n_estimators: int = 100,
                 max_depth: int = 6,
                 learning_rate: float = 0.3,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 gamma: float = 0,
                 reg_alpha: float = 0,
                 reg_lambda: float = 1,
                 min_child_weight: int = 1):
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=seed,
            use_label_encoder=False,
            eval_metric='logloss',
            device='cuda',
        )

    def fit(self, train: FeatureSpace):
        # fit the scaler
        self.scaler.fit(train.features)
        X_scaled = self.scaler.transform(train.features)
        
        # use cupy array directly for GPU
        X = X_scaled
        y = train.labels
        self.model.fit(X, y)


    def predict_proba(self, evaluate: FeatureSpace) -> cp.ndarray:
        X_scaled = self.scaler.transform(evaluate.features)
        proba = self.model.predict_proba(X_scaled)
        return proba
