"""Exact Rashomon-set exploration for ordinary least squares.

The Rashomon set at radius ``r`` is every linear model whose training MSE is
within ``r`` of a reference model's MSE.  For squared error this set is an
ellipsoid with a closed form, which is what makes everything here exact:

   MSE(theta) = MSE(theta_ols) + (theta - theta_ols)' Z'Z (theta - theta_ols) / n

where ``theta = (intercept, slopes)`` and ``Z = [1, X]``.  Setting the right-hand
side equal to ``allowed_loss`` gives the ellipsoid; every query below is a
support-function or Schur-complement calculation on it.

Layout
------
``_OLSGeometry``  pure numpy math; knows nothing about units, names or sklearn.
``Prism``         user-facing: radius bookkeeping, feature names, result shaping.
``Spec``          builds concrete sklearn models at chosen points of the set.

v1 is squared-error only.  A GLM/Laplace path should be a sibling geometry class
with the same four methods, not new branches inside these classes.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from .geometries import _OLSGeometry
from .utils import (
    _abs_interval,
    _as_1d,
    _as_2d,
    check_mse,
    _validate_linear_model,
    _feature_indices,
    _resolve_feature_names,
)

__all__ = ["Prism", "Spec"]

_UNITS = ("percent", "raw")
_MODES = ("both", "min", "max")


class Prism:
    """Explore models within a radius of a reference model's training loss.

    Parameters
    ----------
    model : Fitted sklearn-style linear regressor defining the baseline loss.
    X : Feature matrix the model was fitted on.
    y : Target vector the model was fitted on.
    r : Radius of the Rashomon set (default 1.0).
    units : Interpretation of radius, either "percent" or "raw" (default "percent").
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any,
        r: float = 1.0,
        units: str = "percent",
    ):
        self.X, self.y = _as_2d(X), _as_1d(y)
        if len(self.X) != len(self.y):
            raise ValueError(f"X has {len(self.X)} rows but y has {len(self.y)}.")
        
        _validate_linear_model(model, self.X.shape[1])
        if not getattr(model, "fit_intercept", True):
            raise ValueError("Prism requires a model fitted with fit_intercept=True.")

        self.reference_model = model
        self.feature_names = _resolve_feature_names(model, self.X, self.X.shape[1])

        self._geometry = _OLSGeometry(self.X, self.y)
        ref_pred = self.X @ np.asarray(model.coef_, float).ravel() + float(np.ravel(model.intercept_)[0])
        self.baseline_loss = float(np.mean((self.y - ref_pred) ** 2))
        self.update_r(r, units)

    @classmethod
    def from_data(cls, X: Any, y: Any, r: float = 1.0, units: str = "percent"):
        """Fit an OLS reference model for you, then build the Prism.
        
        Parameters
        ----------
        X : Feature matrix to fit the OLS model on.
        y : Target vector to fit the OLS model on.
        r : Radius of the Rashomon set (default 1.0).
        units : Interpretation of radius, either "percent" or "raw" (default "percent").
        """
        return cls(LinearRegression().fit(X, y), X, y, r, units)

    def update_r(self, r: float, units: Optional[str] = None) -> None:
        """Change the radius.
        
        Parameters
        ----------
        r : New radius value.
        units : Optional new units ("percent" or "raw"); keeps current if None.
        """
        units = getattr(self, "units", "percent") if units is None else units
        if units not in _UNITS:
            raise ValueError(f"units must be one of {_UNITS}, got {units!r}")
        if r < 0:
            raise ValueError("radius must be non-negative")
        self.r, self.units = float(r), units

    @property
    def allowed_loss(self) -> float:
        """Largest training MSE a model may have and still be in the set."""
        if self.units == "percent":
            return self.baseline_loss * (1.0 + self.r / 100.0)
        return self.baseline_loss + self.r

    @property
    def _budget(self) -> float:
        g = self._geometry
        return max(0.0, g.n * (self.allowed_loss - g.min_loss))

    def get_coefs(self) -> Tuple[np.ndarray, List[Any]]:
        """OLS slopes (the centre of the set) and their feature names."""
        return self._geometry.coef.copy(), list(self.feature_names)

    @property
    def coef_(self) -> np.ndarray:
        return self._geometry.coef.copy()

    @property
    def intercept_(self) -> float:
        return self._geometry.intercept

    def coef_bounds(
        self,
        features: Optional[Sequence[str]] = None,
        magnitude: bool = False,
        mode: str = "both",
    ) -> Dict[Any, List[float]]:
        """Range of each requested coefficient across the set.

        Parameters
        ----------
        features : Specific feature names to get bounds for (None for all).
        magnitude : If True, returns bounds for the absolute value of coefficients.
        mode : Return format, one of "both" ([lo, hi]), "min" ([lo]), or "max" ([hi]).
        """
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        idx = range(len(self.feature_names)) if features is None else _feature_indices(features, self.feature_names)
        center = self._geometry.coef
        width = self._geometry.coef_halfwidths(self._budget)

        out: Dict[Any, List[float]] = {}
        for i in idx:
            lo, hi = float(center[i] - width[i]), float(center[i] + width[i])
            if magnitude:
                lo, hi = _abs_interval(lo, hi)
            out[self.feature_names[i]] = {"both": [lo, hi], "min": [lo], "max": [hi]}[mode]
        return out

    def disagreement(self, X: Any = None):
        """Min / OLS / max prediction across the set, per row of ``X``.

        Parameters
        ----------
        X : Data matrix to predict on (defaults to training data).
        """
        if X is None:
            X = self.X
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"X has {X.shape[1]} columns but the model has {len(self.feature_names)}.")

        optimal = self._geometry.predict(X)
        width = self._geometry.prediction_halfwidths(X, self._budget)
        result = {"min": optimal - width, "optimal": optimal, "max": optimal + width}
        return result

    def query_droppable(self, features: Any) -> bool:
        """Can all of ``features`` be removed (others re-fit) and stay in the set?
        
        Parameters
        ----------
        features : List or single feature name to check for removal.
        """
        idx = _feature_indices(features, self.feature_names)
        return self._geometry.loss_with_zeroed(idx) <= self.allowed_loss

    def droppable(self, max_terms: Optional[int] = 1) -> List[List[str]]:
        """Every droppable feature subset of size <= ``max_terms`` (None = no limit).

        Parameters
        ----------
        max_terms : Maximum size of feature subsets to test (None for unlimited).
        """
        p = len(self.feature_names)
        max_terms = p if max_terms is None else min(max_terms, p)
        found: List[List[str]] = []
        previous: set = set()
        for size in range(1, max_terms + 1):
            level = [
                combo
                for combo in combinations(range(p), size)
                if (size == 1 or all(sub in previous for sub in combinations(combo, size - 1)))
                and self._geometry.loss_with_zeroed(combo) <= self.allowed_loss
            ]
            if not level:
                break
            found.extend([self.feature_names[i] for i in combo] for combo in level)
            previous = set(level)
        return found

    def spec(self) -> "Spec":
        """A Spec over the same data, model and feature names."""
        return Spec(self.reference_model, self.X, self.y, feature_names=self.feature_names)


class Spec:
    """Build concrete linear models at chosen points of the loss surface.

    Parameters
    ----------
    model : The base sklearn-style linear model template.
    X : Feature matrix for calculating MSE or refitting.
    y : Target vector for calculating MSE or refitting.
    feature_names : Optional list of string names for columns of X.
    """

    def __init__(self, model: Any, X: Any, y: Any, feature_names: Optional[Sequence[Any]] = None):
        self.X, self.y = _as_2d(X), _as_1d(y)
        _validate_linear_model(model, self.X.shape[1])
        self.model = model
        self.feature_names = (
            list(feature_names) if feature_names is not None
            else _resolve_feature_names(model, X, self.X.shape[1])
        )
        self._fit_intercept = bool(getattr(model, "fit_intercept", True))

    def get_coefs(self) -> Tuple[np.ndarray, List[Any]]:
        """Return the coefficients and feature names of the current model."""
        return np.asarray(self.model.coef_, float).ravel(), list(self.feature_names)

    def mse(self, model: Any = None, X: Any = None, y: Any = None) -> float:
        """Compute a model's MSE.
        
        Parameters
        ----------
        model : Model to evaluate (defaults to this Spec's model).
        X : Features to evaluate on (defaults to this Spec's X).
        y : Targets to evaluate on (defaults to this Spec's y).
        """
        return check_mse(
            self.model if model is None else model,
            self.X if X is None else X,
            self.y if y is None else y,
        )

    def from_coefs(self, coef_dict: dict[str, float]):
        """Least-MSE model with feature(s) fixed at target coefficient(s).
        
        Parameters
        ----------
        coef_dict : Dictionary mapping feature names to fixed target coefficient values.
        """
        return self._fit_with_fixed({k: float(v) for k, v in coef_dict.items()})

    def sans_features(self, features: Any):
        """Least-MSE model with ``features`` fixed at zero.
        
        Parameters
        ----------
        features : List or single feature name to force to zero.
        """
        if isinstance(features, str):
            features = [features]
        return self._fit_with_fixed({f: 0.0 for f in features})

    def _fit_with_fixed(self, fixed: Dict[Any, float]):
        p = len(self.feature_names)
        fixed_idx = _feature_indices(list(fixed), self.feature_names)
        values = np.array([fixed[self.feature_names[i]] for i in fixed_idx], dtype=float)
        free = [i for i in range(p) if i not in fixed_idx]

        target = self.y - self.X[:, fixed_idx] @ values
        design = self.X[:, free]
        if self._fit_intercept:
            design = np.column_stack([np.ones(len(self.y)), design])
        solution = np.linalg.lstsq(design, target, rcond=None)[0] if design.shape[1] else np.empty(0)

        intercept = 0.0
        if self._fit_intercept:
            intercept, solution = float(solution[0]), solution[1:]
        coefs = np.zeros(p)
        coefs[fixed_idx] = values
        coefs[free] = solution
        return self._build_model(coefs, intercept)

    def _build_model(self, coefs: np.ndarray, intercept: float):
        model = clone(self.model)
        model.coef_ = coefs
        model.intercept_ = intercept
        model.n_features_in_ = len(coefs)
        if hasattr(self.model, "feature_names_in_"):
            model.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
        return model