import cupy as cp
from cuml import LogisticRegression
from cuml.preprocessing import StandardScaler

from hproj.classifiers.classifier import Classifier, ClassifierFactory
from hproj.data.feature_space import FeatureSpace


@ClassifierFactory.register('logistic-regression')
class LogisticRegressionClassifier(Classifier):
    def __init__(self, seed, C):
        # self.seed = logistic regression has a convex loss surface so no seed required
        self.scaler = StandardScaler()
        self.C = C

    def fit(self, train: FeatureSpace):
        # fit the scalar
        self.scaler.fit(train.features)
        X = self.scaler.transform(train.features)

        # fit the logistic regression model
        self.lg = LogisticRegression(C=self.C)
        self.lg.fit(X, train.labels)

    def predict_proba(self, evaluate: FeatureSpace) -> cp.ndarray:
        X = self.scaler.transform(evaluate.features)
        return self.lg.predict_proba(X)    
