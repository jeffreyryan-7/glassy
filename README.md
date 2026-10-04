# glassy

`glassy` explores predictive multiplicity in linear regression: how much model
coefficients and predictions can vary while training error stays close to a
reference model's error. For squared-error loss, it computes exact ranges over
this set of models.

## Install

Install from PyPI:

```bash
python -m pip install glassy
```

## Quick start

`Prism` and `Spec` are separate objects. `Prism` analyzes the model set;
`Spec` builds concrete models. Initialize both with a fitted linear regressor
and the training data:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from glassy import Prism, Spec

rng = np.random.default_rng(7)
X = rng.normal(size=(100, 3))
y = 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.5, size=100)

model = LinearRegression().fit(X, y)

# Include models with training MSE up to 5% above the reference model's MSE.
prism = Prism(model, X, y, r=5.0, units="percent")
spec = Spec(model, X, y)

print(prism.coef_bounds())
predictions = prism.disagreement(X)
print(predictions["min"], predictions["optimal"], predictions["max"])
print(prism.query_droppable("feature_2"))

candidate = spec.sans_features("feature_2")
print(candidate.predict(X)[:5])
```

`Prism.from_data(X, y, r=..., units=...)` is a convenience constructor that
fits an ordinary least squares reference model for you. Radius units are
`"percent"` (relative to the reference model's training MSE) or `"raw"` (an
additive MSE increase).

## API overview

- `prism.coef_bounds(...)` returns coefficient intervals. Set `magnitude=True`
	for ranges of absolute coefficient values. `mode` can be `"both"`, `"min"`,
	or `"max"`; `features` limits the result to selected names.
- `prism.disagreement(X=None)` returns `min`, `optimal`, and `max` prediction
	arrays; it uses the training data when `X` is omitted.
- `prism.query_droppable(features)` checks whether features can be removed
	within the loss limit. `prism.droppable(max_terms=1)` lists removable subsets
	up to that size; use `max_terms=None` to search all subset sizes.
- `prism.update_r(r, units=None)` changes the radius.
- `spec = Spec(model, X, y, feature_names=None)` creates a model-building
	object independently of `Prism`. `spec.sans_features(features)` returns a
	fitted sklearn model with the named coefficients fixed at zero and the
	others refit.
- `spec.from_coefs({"feature_0": value})` returns a fitted model with the
	specified coefficients fixed and the others refit to minimize training MSE.
- `spec.mse(model=None, X=None, y=None)` computes training or supplied-data
	MSE; omitted arguments default to the model and data held by `spec`.
- `prism.spec()` is an optional convenience method that creates a `Spec` from
	the reference model and data already held by that `Prism`.

For NumPy arrays, feature names default to `feature_0`, `feature_1`, and so
on. DataFrame column names are preserved.

## Scope and requirements

`glassy` currently supports numeric linear-regression features, an intercept,
and squared-error loss. The feature matrix must have full column rank;
redundant or constant columns make the model set unbounded and are rejected.
Runtime dependencies are NumPy and scikit-learn.

## Development

Install test dependencies and run the tests from a repository checkout:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Example notebooks are in [`demos/`](demos/).
