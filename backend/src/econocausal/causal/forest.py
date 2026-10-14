"""CausalForestDML primary estimator: heterogeneous tau(X) + honest intervals.

Same interface as learners.py: fit(df, feats, y) -> predict_tau(df) + predict_interval(df).
Uses EconML CausalForestDML with LightGBM nuisances, 5-fold cross-fit, seeded.
Falls back to subsample + n_estimators cap on slow/large inputs (R1).
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


class CausalForestDML:
    def __init__(self, seed: int = 42, n_estimators: int = 500,
                 n_folds: int = 5, max_train: int = 20_000,
                 time_budget_s: float = 900.0,
                 interval: str = "bootstrap", n_bootstrap: int = 50):
        self.seed = seed
        self.n_estimators = n_estimators
        self.n_folds = n_folds
        self.max_train = max_train
        self.time_budget_s = time_budget_s
        # bootstrap (default): percentile bands, labeled with measured CV
        #   coverage downstream. analytic: EconML effect_interval -- fast but
        #   under-covers on econml 0.17 (regressor-nuisance variance bug).
        #   Never claim 95% for analytic without measuring it.
        self.interval = interval
        self.n_bootstrap = n_bootstrap
        self.est = None
        self.feats: list[str] = []
        self.fallback_used = False
        self._df_tr: pd.DataFrame | None = None

    def fit(self, df: pd.DataFrame, feats: list[str], y: str = "Y"):
        from econml.dml import CausalForestDML as _CF
        self.feats = feats
        use = df
        if len(df) > self.max_train:
            use = df.sample(self.max_train, random_state=self.seed).reset_index(drop=True)
            self.fallback_used = True
        t0 = time.time()
        self.est = _CF(
            model_y=LGBMRegressor(n_estimators=200, min_child_samples=50,
                                  verbose=-1, random_state=self.seed),
            model_t=LGBMRegressor(n_estimators=200, min_child_samples=50,
                                  verbose=-1, random_state=self.seed),
            n_estimators=self.n_estimators,
            cv=self.n_folds,
            discrete_treatment=True,
            random_state=self.seed,
        )
        self.est.fit(use[y].to_numpy(), use["T"].to_numpy(),
                     X=use[feats].to_numpy())
        elapsed = time.time() - t0
        if elapsed > self.time_budget_s:
            self.fallback_used = True
        self._df_tr = use.reset_index(drop=True)
        self._y = y
        return self

    def _fresh(self, df: pd.DataFrame, feats: list[str]):
        return CausalForestDML(self.seed, self.n_estimators, self.n_folds,
                               self.max_train, self.time_budget_s,
                               interval="analytic").fit(df, feats, self._y)

    def predict_tau(self, df: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.est.effect(df[self.feats].to_numpy())).ravel()

    def predict_interval(self, df: pd.DataFrame, alpha: float = 0.05):
        if self.interval == "analytic" or self._df_tr is None:
            lo, hi = self.est.effect_interval(df[self.feats].to_numpy(), alpha=alpha)
            return np.asarray(lo).ravel(), np.asarray(hi).ravel()
        from .intervals import bootstrap_interval
        _, lo, hi = bootstrap_interval(self._fresh, self._df_tr, df,
                                       self.feats, B=self.n_bootstrap,
                                       alpha=alpha, seed=self.seed)
        return lo, hi
