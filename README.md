# glassy

`glassy` enables exploration of preditive multiplicity in linear regression models

## Install

From a clone of this repository, install the package and its runtime
dependencies with:

```bash
python -m pip install .
```

## Scope and requirements

This release supports numeric linear-regression features with an intercept and
squared-error loss. The feature matrix must have full column rank; redundant or
constant columns make the set unbounded and are rejected. Runtime dependencies
are NumPy and scikit-learn.

## Development

Install the test dependencies and run the test suite with:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The `demos/` directory contains tutorial notebooks.
