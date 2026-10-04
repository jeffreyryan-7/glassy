from typing import Any, List, Sequence, Tuple

import numpy as np


def _as_2d(X: Any) -> np.ndarray:
	X = np.asarray(X, dtype=float)
	return X.reshape(-1, 1) if X.ndim == 1 else X


def _as_1d(y: Any) -> np.ndarray:
	return np.asarray(y, dtype=float).ravel()


def check_mse(model: Any, X: Any, y: Any) -> float:
	"""Mean squared error for a fitted linear model and matching data."""
	if hasattr(X, "columns") and getattr(model, "feature_names_in_", None) is not None:
		X = X.loc[:, model.feature_names_in_]
	X_array, y_array = _as_2d(X), _as_1d(y)
	if X_array.shape[1] != np.asarray(model.coef_).size:
		raise ValueError("X column count must match the model coefficients.")
	if len(X_array) != len(y_array):
		raise ValueError(f"X has {len(X_array)} rows but y has {len(y_array)}.")
	predictions = X_array @ np.asarray(model.coef_, dtype=float).ravel()
	predictions += float(np.ravel(model.intercept_)[0])
	return float(np.mean((y_array - predictions) ** 2))


def _resolve_feature_names(model: Any, X: Any, n_features: int) -> List[Any]:
	"""Names from the fitted model, else DataFrame columns, else ``feature_i``."""
	names = getattr(model, "feature_names_in_", None)
	if names is None and hasattr(X, "columns"):
		names = X.columns
	if names is None:
		return [f"feature_{i}" for i in range(n_features)]
	names = list(names)
	if len(names) != n_features:
		raise ValueError(f"Got {len(names)} feature names for {n_features} columns.")
	return names


def _feature_indices(features: Any, names: Sequence[Any]) -> List[int]:
	"""Column indices for one name or a sequence of names (deduplicated, in order)."""
	if isinstance(features, str):
		features = [features]
	unknown = [f for f in features if f not in names]
	if unknown:
		raise ValueError(f"Unknown feature(s): {unknown}")
	return list(dict.fromkeys(list(names).index(f) for f in features))


def _validate_linear_model(model: Any, n_features: int) -> None:
	missing = [a for a in ("coef_", "intercept_") if not hasattr(model, a)]
	if missing:
		raise TypeError(f"`model` must be a fitted linear regressor; missing {missing}.")
	if np.asarray(model.coef_).size != n_features:
		raise ValueError(
			f"model has {np.asarray(model.coef_).size} coefficients but X has {n_features} columns."
		)


def _abs_interval(lo: float, hi: float) -> Tuple[float, float]:
	"""Range of |x| for x in [lo, hi]."""
	if lo <= 0.0 <= hi:
		return 0.0, max(-lo, hi)
	return tuple(sorted((abs(lo), abs(hi))))
