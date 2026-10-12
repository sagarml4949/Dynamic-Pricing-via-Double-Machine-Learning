"""Baselines behind one interface: fit(Y,T,X) -> tau_hat.

S/T/X-learners (LightGBM) + LinearDML. CausalForestDML lands in Week 2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor


def _x(df: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return df[feats].to_numpy()


class SLearner:
    def __init__(self, seed: int = 42):
        self.m = LGBMClassifier(n_estimators=200, num_leaves=31,
                                min_child_samples=50, verbose=-1, random_state=seed)

    def fit(self, df: pd.DataFrame, feats: list[str], y: str = "Y"):
        Xt = np.column_stack([_x(df, feats), df["T"].to_numpy()])
        self.feats = feats
        self.m.fit(Xt, df[y].to_numpy())
        return self

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        X = _x(df, self.feats)
        return self.m.predict_proba(np.column_stack([X, np.ones(len(X))]))[:, 1] - \
            self.m.predict_proba(np.column_stack([X, np.zeros(len(X))]))[:, 1]


class TLearner:
    def __init__(self, seed: int = 42):
        kw = dict(n_estimators=200, num_leaves=31, min_child_samples=50,
                  verbose=-1, random_state=seed)
        self.m1, self.m0 = LGBMClassifier(**kw), LGBMClassifier(**kw)

    def fit(self, df: pd.DataFrame, feats: list[str], y: str = "Y"):
        self.feats = feats
        d1, d0 = df[df["T"] == 1], df[df["T"] == 0]
        self.m1.fit(_x(d1, feats), d1[y].to_numpy())
        self.m0.fit(_x(d0, feats), d0[y].to_numpy())
        return self

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        X = _x(df, self.feats)
        return self.m1.predict_proba(X)[:, 1] - self.m0.predict_proba(X)[:, 1]


class XLearner:
    """X-learner with propensity-weighted blend of the two imputed effects."""

    def __init__(self, seed: int = 42):
        kw = dict(n_estimators=200, num_leaves=31, min_child_samples=50,
                  verbose=-1, random_state=seed)
        self.m1, self.m0 = LGBMClassifier(**kw), LGBMClassifier(**kw)
        self.t1 = LGBMRegressor(verbose=-1, random_state=seed)
        self.t0 = LGBMRegressor(verbose=-1, random_state=seed)
        self.prop = LGBMClassifier(n_estimators=100, verbose=-1, random_state=seed)

    def fit(self, df: pd.DataFrame, feats: list[str], y: str = "Y"):
        self.feats = feats
        X, T, Y = _x(df, feats), df["T"].to_numpy(), df[y].to_numpy()
        d1, d0 = T == 1, T == 0
        self.m1.fit(X[d1], Y[d1])
        self.m0.fit(X[d0], Y[d0])
        self.prop.fit(X, T)
        mu1, mu0 = self.m1.predict_proba(X)[:, 1], self.m0.predict_proba(X)[:, 1]
        self.t1.fit(X[d1], Y[d1] - mu0[d1])  # treated: Y1 - mu0
        self.t0.fit(X[d0], mu1[d0] - Y[d0])  # control: mu1 - Y0
        return self

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        X = _x(df, self.feats)
        g = self.prop.predict_proba(X)[:, 1]
        return g * self.t0.predict(X) + (1 - g) * self.t1.predict(X)


class LinearDMLBaseline:
    """EconML LinearDML with LightGBM nuisances. Fast CI-capable baseline."""

    def __init__(self, seed: int = 42, n_folds: int = 5):
        self.seed = seed
        self.n_folds = n_folds
        self.est = None
        self.feats: list[str] = []

    def fit(self, df: pd.DataFrame, feats: list[str], y: str = "Y"):
        from econml.dml import LinearDML
        self.feats = feats
        self.est = LinearDML(
            model_y=LGBMRegressor(n_estimators=200, min_child_samples=50,
                                  verbose=-1, random_state=self.seed),
            model_t=LGBMRegressor(n_estimators=200, min_child_samples=50,
                                  verbose=-1, random_state=self.seed),
            cv=self.n_folds, random_state=self.seed,
            discrete_treatment=True,
        )
        self.est.fit(df[y].to_numpy(), df["T"].to_numpy(),
                     X=df[feats].to_numpy())
        return self

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        return self.est.effect(df[self.feats].to_numpy())
