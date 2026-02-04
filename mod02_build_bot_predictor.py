# packages
from sklearn.ensemble import GradientBoostingClassifier

# set seed
seed = 314


def train_model(X, y, seed=seed):
    """
    Build a GBM on given data
    """
    model = GradientBoostingClassifier(
        learning_rate=0.0025,
        n_estimators=1400,
        max_depth=2,
        subsample=1,
        min_samples_leaf=1,
        random_state=seed,
    )
    model.fit(X, y)
    return model
