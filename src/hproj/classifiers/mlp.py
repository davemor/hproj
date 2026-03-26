import cupy as cp
from cuml import LogisticRegression
from cuml.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from hproj.classifiers.classifier import Classifier, ClassifierFactory
from hproj.data.feature_space import FeatureSpace


@ClassifierFactory.register('mlp')
class MLPClassifier(Classifier):
    def __init__(self, seed, 
                 hidden_layer_sizes: tuple[int, ...] = (100,),
                 activation: str = 'relu',
                 solver: str = 'adam',
                 alpha: float = 0.0001,
                 learning_rate: str = 'constant',
                 learning_rate_init: float = 0.001,
                 max_iter: int = 200,
                 early_stopping: bool = False):
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            random_state=seed,
        )

    def fit(self, train: FeatureSpace):
        # fit the scaler
        self.scaler.fit(train.features)
        X_scaled = self.scaler.transform(train.features)

        # convert to numpy for sklearn
        X_np = X_scaled.asnumpy()
        y_np = train.labels.asnumpy()

        # fit the model
        self.model.fit(X_np, y_np)

    def predict_proba(self, evaluate: FeatureSpace) -> cp.ndarray:
        # scale the input space
        X_scaled = self.scaler.transform(evaluate.features)
        X_np = X_scaled.asnumpy()

        # predict using the model
        proba_np = self.model.predict_proba(X_np)

        # convert back to cupy
        return cp.asarray(proba_np)