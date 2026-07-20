"""
Four simple, explainable models for predicting short-horizon price change from
a single order-book snapshot (44 features in -> one number out):

  linear   scikit-learn LinearRegression     (fit a straight-line relationship)
  knn      scikit-learn KNeighborsRegressor   (average the most similar snapshots)
  tree     scikit-learn DecisionTreeRegressor (a flowchart of yes/no rules)
  mlp      a small PyTorch neural network     (the one deep-learning baseline)

The three classic models are one line of scikit-learn each. MLP is the only
neural network here -- two hidden layers, nothing fancier.
"""
import torch
from torch import nn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

# name -> how to build that scikit-learn model
SKLEARN_MODELS = {
    'linear': lambda: LinearRegression(),
    'knn': lambda: KNeighborsRegressor(n_neighbors=25),
    'tree': lambda: DecisionTreeRegressor(max_depth=6, random_state=222),
}


def build_sklearn_model(name):
    if name not in SKLEARN_MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {list(SKLEARN_MODELS)}")
    return SKLEARN_MODELS[name]()


class MLP(nn.Module):
    """A small fully-connected network: 44 features -> 128 -> 64 -> 1."""

    def __init__(self, in_features=44, out_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)
