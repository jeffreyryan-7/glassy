"""Mathematical geometry for supported loss surfaces."""
from typing import Sequence

import numpy as np


class _OLSGeometry:
	"""Quadratic structure of the squared-error loss around its OLS minimiser.

	Everything takes a ``budget``: the allowed increase in *sum* of squared errors,
	``n * (allowed_loss - min_loss)``. Callers own the unit conversion.
	"""

	def __init__(self, X: np.ndarray, y: np.ndarray):
		n, p = X.shape
		self.n = n
		self.x_mean = X.mean(axis=0)
		Xc = X - self.x_mean
		yc = y - y.mean()
		if np.linalg.matrix_rank(Xc) < p:
			raise ValueError(
				"X_train has linearly dependent columns (including constant columns), "
				"so the Rashomon set is unbounded. Drop or combine the redundant features."
			)
		self.inv_gram = np.linalg.inv(Xc.T @ Xc)
		self.coef = self.inv_gram @ (Xc.T @ yc)
		self.intercept = float(y.mean() - self.x_mean @ self.coef)
		self.min_loss = float(np.mean((yc - Xc @ self.coef) ** 2))

	def predict(self, X: np.ndarray) -> np.ndarray:
		return X @ self.coef + self.intercept

	def coef_halfwidths(self, budget: float) -> np.ndarray:
		"""Half-width of each slope's range over the set."""
		return np.sqrt(budget * np.diag(self.inv_gram))

	def prediction_halfwidths(self, X: np.ndarray, budget: float) -> np.ndarray:
		"""Half-width of the prediction range at each row of X.

		Uses the full (intercept + slopes) ellipsoid, so the leverage is
		``1/n + (x - x_mean)' (Xc'Xc)^-1 (x - x_mean)``.
		"""
		Xc = X - self.x_mean
		leverage = 1.0 / self.n + np.einsum("ij,jk,ik->i", Xc, self.inv_gram, Xc)
		return np.sqrt(budget * leverage)

	def loss_with_zeroed(self, idx: Sequence[int]) -> float:
		"""Smallest MSE achievable with the slopes at ``idx`` forced to zero.

		Others (and the intercept) are free to re-fit, so this is exactly the loss
		of ``Spec.sans_features``. Monotone in ``idx``: dropping more never helps.
		"""
		idx = list(idx)
		if not idx:
			return self.min_loss
		beta = self.coef[idx]
		block = self.inv_gram[np.ix_(idx, idx)]
		return self.min_loss + float(beta @ np.linalg.solve(block, beta)) / self.n
