"""Tests for glassy's Prism / Spec (OLS, squared-error Rashomon sets).

Organisation
------------
1. Basic functionality   - construction, shapes, defaults, round trips.
2. Mathematical truth    - every closed-form claim is re-derived here with
                           *independent* code (explicit Z'Z, Cholesky sampling,
                           brute-force refits via sklearn) and compared.
3. Invariances/properties- scaling, row permutation, monotonicity in r.
4. Input validation.

The math tests never call ``_OLSGeometry`` directly, so they check the public
behaviour against first principles rather than against the implementation.

Notation: Z = [1, X], A = Z'Z, theta = (intercept, slopes), n = rows,
    MSE(theta) = MSE_ols + (theta - theta_ols)' A (theta - theta_ols) / n
    budget     = n * (allowed_loss - MSE_ols)          (ellipsoid: d'Ad <= budget)
"""
from itertools import combinations

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from glassy import Prism, Spec

RTOL = 1e-8
ATOL = 1e-10
SEEDS = [0, 1, 2]


# --------------------------------------------------------------------------- #
# Helpers (independent reference implementations)
# --------------------------------------------------------------------------- #
def make_data(seed=0, n=300, p=5, noise=1.0):
    """Correlated features; x_last is pure noise, x_{p-2} is weak."""
    rng = np.random.default_rng(seed)
    mix = np.eye(p) + 0.4 * rng.standard_normal((p, p))
    X = rng.standard_normal((n, p)) @ mix + rng.uniform(-3, 3, size=p)
    beta = np.linspace(2.0, -1.0, p)
    beta[-2] = 0.15
    beta[-1] = 0.0
    y = 1.5 + X @ beta + noise * rng.standard_normal(n)
    return X, y


def design(X):
    return np.column_stack([np.ones(len(X)), X])


def mse(theta, X, y):
    return float(np.mean((y - design(X) @ theta) ** 2))


def ols_theta(X, y):
    return np.linalg.lstsq(design(X), y, rcond=None)[0]


class Ref:
    """Everything we need to know about the ellipsoid, computed from scratch."""

    def __init__(self, prism, X, y):
        self.X, self.y = X, y
        self.n = len(y)
        Z = design(X)
        self.A = Z.T @ Z
        self.Ainv = np.linalg.inv(self.A)
        self.theta = ols_theta(X, y)
        self.min_loss = mse(self.theta, X, y)
        self.allowed = prism.allowed_loss
        self.budget = max(0.0, self.n * (self.allowed - self.min_loss))

    def quad(self, theta):
        d = theta - self.theta
        return float(d @ self.A @ d)

    def sphere_point(self, u):
        """Map unit vector u to a boundary point of the ellipsoid."""
        L = np.linalg.cholesky(self.A)
        return self.theta + np.sqrt(self.budget) * np.linalg.solve(L.T, u)

    def sample_boundary(self, k, rng):
        U = rng.standard_normal((k, len(self.theta)))
        U /= np.linalg.norm(U, axis=1, keepdims=True)
        return np.array([self.sphere_point(u) for u in U])

    def coef_halfwidth(self):
        return np.sqrt(self.budget * np.diag(self.Ainv))[1:]

    def pred_halfwidth(self, Xnew):
        Zn = design(np.atleast_2d(Xnew))
        return np.sqrt(self.budget * np.einsum("ij,jk,ik->i", Zn, self.Ainv, Zn))


def subset_losses(X, y):
    """Training MSE after dropping each non-empty feature subset (brute force)."""
    p = X.shape[1]
    out = {}
    for size in range(1, p + 1):
        for combo in combinations(range(p), size):
            keep = [j for j in range(p) if j not in combo]
            if keep:
                m = LinearRegression().fit(X[:, keep], y)
                out[frozenset(combo)] = float(np.mean((y - m.predict(X[:, keep])) ** 2))
            else:
                out[frozenset(combo)] = float(np.var(y))
    return out


def as_sets(subsets, names):
    pos = {n: i for i, n in enumerate(names)}
    return {frozenset(pos[f] for f in s) for s in subsets}


@pytest.fixture(params=SEEDS)
def setup(request):
    X, y = make_data(seed=request.param)
    prism = Prism.from_data(X, y, r=2.0, units="percent")
    return prism, X, y, Ref(prism, X, y)


@pytest.fixture
def data():
    return make_data(seed=0)


# --------------------------------------------------------------------------- #
# 1. Basic functionality
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_from_data_matches_explicit_constructor(self, data):
        X, y = data
        a = Prism.from_data(X, y, r=3.0)
        b = Prism(LinearRegression().fit(X, y), X, y, r=3.0)
        assert a.allowed_loss == pytest.approx(b.allowed_loss, rel=RTOL)
        np.testing.assert_allclose(a.coef_, b.coef_, rtol=RTOL)
        assert a.coef_bounds() == pytest.approx(b.coef_bounds())

    def test_coef_and_intercept_match_sklearn(self, data):
        X, y = data
        m = LinearRegression().fit(X, y)
        p = Prism(m, X, y)
        np.testing.assert_allclose(p.coef_, m.coef_, rtol=1e-7, atol=1e-9)
        assert p.intercept_ == pytest.approx(m.intercept_, rel=1e-7)

    def test_get_coefs_returns_copy_and_names(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        coefs, names = p.get_coefs()
        assert len(coefs) == len(names) == X.shape[1]
        coefs[:] = 0.0  # mutating the result must not corrupt the Prism
        assert np.any(p.coef_ != 0.0)
        p.coef_[:] = 0.0
        assert np.any(p.coef_ != 0.0)

    def test_baseline_loss_is_reference_training_mse(self, data):
        X, y = data
        m = Ridge(alpha=50.0).fit(X, y)
        p = Prism(m, X, y)
        assert p.baseline_loss == pytest.approx(np.mean((y - m.predict(X)) ** 2), rel=RTOL)

    def test_pandas_feature_names(self, data):
        pd = pytest.importorskip("pandas")
        X, y = data
        cols = [f"feat_{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
        p = Prism(LinearRegression().fit(df, y), df, y)
        assert list(p.feature_names) == cols
        assert list(p.coef_bounds()) == cols
        assert p.get_coefs()[1] == cols

    def test_spec_handoff_shares_state(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        s = p.spec()
        assert isinstance(s, Spec)
        assert s.feature_names == list(p.feature_names)
        np.testing.assert_array_equal(s.X, p.X)
        assert s.model is p.reference_model


class TestRadius:
    def test_percent_units(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=10.0, units="percent")
        assert p.allowed_loss == pytest.approx(p.baseline_loss * 1.10, rel=RTOL)

    def test_raw_units(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=0.25, units="raw")
        assert p.allowed_loss == pytest.approx(p.baseline_loss + 0.25, rel=RTOL)

    def test_update_r_changes_bounds_and_keeps_units(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=1.0, units="raw")
        narrow = p.coef_bounds()
        p.update_r(4.0)  # units=None keeps "raw"
        assert p.units == "raw" and p.r == 4.0
        assert p.allowed_loss == pytest.approx(p.baseline_loss + 4.0)
        k = p.feature_names[0]
        assert p.coef_bounds()[k][1] > narrow[k][1]

    def test_update_r_can_switch_units(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=1.0, units="percent")
        p.update_r(0.5, units="raw")
        assert p.units == "raw"
        assert p.allowed_loss == pytest.approx(p.baseline_loss + 0.5)

    def test_percent_and_raw_equivalent_when_matched(self, data):
        X, y = data
        a = Prism.from_data(X, y, r=5.0, units="percent")
        raw = a.baseline_loss * 0.05
        b = Prism.from_data(X, y, r=raw, units="raw")
        for k in a.feature_names:
            np.testing.assert_allclose(a.coef_bounds()[k], b.coef_bounds()[k], rtol=1e-9)


class TestCoefBoundsApi:
    def test_default_covers_all_features_with_lo_le_center_le_hi(self, setup):
        prism, X, y, ref = setup
        b = prism.coef_bounds()
        assert list(b) == list(prism.feature_names)
        for i, k in enumerate(prism.feature_names):
            lo, hi = b[k]
            assert lo <= prism.coef_[i] <= hi

    def test_feature_subset(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        names = list(p.feature_names)
        sub = p.coef_bounds(features=[names[2], names[0]])
        assert set(sub) == {names[0], names[2]}
        assert sub[names[0]] == pytest.approx(p.coef_bounds()[names[0]])

    def test_modes(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        both, lo, hi = (p.coef_bounds(mode=m) for m in ("both", "min", "max"))
        for k in p.feature_names:
            assert lo[k] == [both[k][0]]
            assert hi[k] == [both[k][1]]

    def test_magnitude_interval_rules(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=5.0)
        raw, mag = p.coef_bounds(), p.coef_bounds(magnitude=True)
        for k in p.feature_names:
            lo, hi = raw[k]
            mlo, mhi = mag[k]
            assert mhi == pytest.approx(max(abs(lo), abs(hi)))
            if lo <= 0 <= hi:
                assert mlo == pytest.approx(0.0)
            else:
                assert mlo == pytest.approx(min(abs(lo), abs(hi)))
            assert 0 <= mlo <= mhi


class TestDisagreementApi:
    def test_default_is_training_data_and_ordered(self, setup):
        prism, X, y, ref = setup
        d = prism.disagreement()
        assert set(d) == {"min", "optimal", "max"}
        for v in d.values():
            assert v.shape == (len(y),)
        assert np.all(d["min"] <= d["optimal"] + ATOL)
        assert np.all(d["optimal"] <= d["max"] + ATOL)

    def test_optimal_equals_ols_prediction(self, setup):
        prism, X, y, ref = setup
        np.testing.assert_allclose(
            prism.disagreement()["optimal"],
            LinearRegression().fit(X, y).predict(X),
            rtol=1e-7, atol=1e-8,
        )

    def test_new_rows_and_single_row(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        rng = np.random.default_rng(9)
        Xn = rng.standard_normal((7, X.shape[1]))
        d = p.disagreement(Xn)
        assert d["optimal"].shape == (7,)
        one = p.disagreement(Xn[0])  # 1-D -> single row
        assert one["optimal"].shape == (1,)
        assert one["max"][0] == pytest.approx(d["max"][0])
        assert one["min"][0] == pytest.approx(d["min"][0])


class TestSpecApi:
    def test_builders_return_same_class_full_width(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        names = s.feature_names
        for m in (s.sans_features(names[1]), s.from_coefs({names[0]: 0.5})):
            assert type(m) is type(s.model)
            assert m.coef_.shape == (X.shape[1],)
            assert m.predict(X).shape == (len(y),)
            assert m.n_features_in_ == X.shape[1]

    def test_does_not_mutate_reference_model(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        before = s.model.coef_.copy(), s.model.intercept_
        s.sans_features(s.feature_names[0])
        s.from_coefs({s.feature_names[1]: 3.0})
        np.testing.assert_array_equal(s.model.coef_, before[0])
        assert s.model.intercept_ == before[1]

    def test_sans_features_accepts_str_or_list(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        k = s.feature_names[1]
        np.testing.assert_allclose(s.sans_features(k).coef_, s.sans_features([k]).coef_)

    def test_mse_defaults_and_overrides(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        assert s.mse() == pytest.approx(np.mean((y - s.model.predict(X)) ** 2), rel=RTOL)
        m = s.sans_features(s.feature_names[0])
        assert s.mse(m) == pytest.approx(np.mean((y - m.predict(X)) ** 2), rel=RTOL)
        assert s.mse(m, X[:50], y[:50]) == pytest.approx(
            np.mean((y[:50] - m.predict(X[:50])) ** 2), rel=RTOL
        )


# --------------------------------------------------------------------------- #
# 2. Mathematical correctness
# --------------------------------------------------------------------------- #
class TestEllipsoidIdentity:
    """The master formula everything else rests on."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_mse_decomposition(self, seed):
        X, y = make_data(seed)
        ref = Ref(Prism.from_data(X, y), X, y)
        rng = np.random.default_rng(seed + 100)
        for _ in range(25):
            theta = ref.theta + rng.standard_normal(len(ref.theta)) * rng.uniform(0, 2)
            lhs = mse(theta, X, y)
            rhs = ref.min_loss + ref.quad(theta) / ref.n
            assert lhs == pytest.approx(rhs, rel=1e-9)

    def test_ols_is_the_minimiser(self, setup):
        prism, X, y, ref = setup
        rng = np.random.default_rng(5)
        for _ in range(25):
            theta = ref.theta + 1e-3 * rng.standard_normal(len(ref.theta))
            assert mse(theta, X, y) >= ref.min_loss

    def test_boundary_points_have_exactly_allowed_loss(self, setup):
        prism, X, y, ref = setup
        rng = np.random.default_rng(1)
        for theta in ref.sample_boundary(30, rng):
            assert mse(theta, X, y) == pytest.approx(ref.allowed, rel=1e-9)

    def test_non_ols_reference_sits_inside_set_at_r0(self, data):
        """Ridge is not the minimiser; with r=0 it must lie exactly on the boundary."""
        X, y = data
        m = Ridge(alpha=200.0).fit(X, y)
        p = Prism(m, X, y, r=0.0, units="raw")
        ref = Ref(p, X, y)
        theta = np.r_[m.intercept_, m.coef_]
        assert ref.budget > 0                      # set is wider than a point
        assert ref.quad(theta) == pytest.approx(ref.budget, rel=1e-8)
        for i, k in enumerate(p.feature_names):
            lo, hi = p.coef_bounds()[k]
            assert lo - 1e-9 <= m.coef_[i] <= hi + 1e-9
        # ellipsoid is still centred on OLS, not on the reference
        np.testing.assert_allclose(p.coef_, ols_theta(X, y)[1:], rtol=1e-7, atol=1e-9)


class TestCoefBoundsMath:
    def test_halfwidth_closed_form(self, setup):
        prism, X, y, ref = setup
        hw = ref.coef_halfwidth()
        b = prism.coef_bounds()
        for i, k in enumerate(prism.feature_names):
            c = ref.theta[i + 1]
            np.testing.assert_allclose(b[k], [c - hw[i], c + hw[i]], rtol=1e-8, atol=1e-10)

    def test_bounds_are_attained_by_a_set_member(self, setup):
        """Support-function extremiser: d = +-sqrt(budget/Ainv_ii) * Ainv[:, i]."""
        prism, X, y, ref = setup
        b = prism.coef_bounds()
        for i, k in enumerate(prism.feature_names):
            j = i + 1
            for sign, end in ((-1, 0), (1, 1)):
                d = sign * np.sqrt(ref.budget / ref.Ainv[j, j]) * ref.Ainv[:, j]
                theta = ref.theta + d
                assert mse(theta, X, y) == pytest.approx(ref.allowed, rel=1e-9)
                assert theta[j] == pytest.approx(b[k][end], rel=1e-8, abs=1e-10)

    def test_no_set_member_exceeds_bounds(self, setup):
        prism, X, y, ref = setup
        b = prism.coef_bounds()
        rng = np.random.default_rng(3)
        pts = ref.sample_boundary(400, rng)
        # also interior points
        radii = rng.uniform(0, 1, size=(400, 1)) ** (1 / len(ref.theta))
        pts = np.vstack([pts, ref.theta + radii * (ref.sample_boundary(400, rng) - ref.theta)])
        for i, k in enumerate(prism.feature_names):
            lo, hi = b[k]
            assert pts[:, i + 1].min() >= lo - 1e-9
            assert pts[:, i + 1].max() <= hi + 1e-9

    def test_rejection_sampling_from_box_agrees(self):
        """Model-free check in 2-D: any theta with loss <= allowed has every
        coefficient inside the reported bounds, and the box is not vacuous."""
        X, y = make_data(0, n=200, p=2)
        p = Prism.from_data(X, y, r=3.0)
        ref = Ref(p, X, y)
        b = p.coef_bounds()
        rng = np.random.default_rng(11)
        hw = ref.coef_halfwidth()
        hw_int = np.sqrt(ref.budget * ref.Ainv[0, 0])
        scale = np.r_[hw_int, hw] * 1.3
        inside = 0
        for _ in range(5000):
            theta = ref.theta + rng.uniform(-1, 1, 3) * scale
            if mse(theta, X, y) <= p.allowed_loss:
                inside += 1
                for i, k in enumerate(p.feature_names):
                    assert b[k][0] - 1e-9 <= theta[i + 1] <= b[k][1] + 1e-9
        assert inside > 100

    def test_r_zero_collapses_to_ols(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=0.0)
        for i, k in enumerate(p.feature_names):
            lo, hi = p.coef_bounds()[k]
            assert lo == pytest.approx(p.coef_[i], abs=1e-6)
            assert hi == pytest.approx(p.coef_[i], abs=1e-6)

    def test_halfwidth_scales_with_sqrt_of_radius(self, data):
        """For an OLS reference budget = n*r (raw), so width ~ sqrt(r)."""
        X, y = data
        p = Prism.from_data(X, y, r=0.1, units="raw")
        w1 = {k: v[1] - v[0] for k, v in p.coef_bounds().items()}
        p.update_r(0.4)
        w2 = {k: v[1] - v[0] for k, v in p.coef_bounds().items()}
        for k in w1:
            assert w2[k] / w1[k] == pytest.approx(2.0, rel=1e-8)

    def test_widths_monotone_in_radius(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=0.5)
        prev = None
        for r in (0.5, 1, 2, 5, 10, 50):
            p.update_r(r)
            w = np.array([hi - lo for lo, hi in p.coef_bounds().values()])
            if prev is not None:
                assert np.all(w >= prev)
            prev = w

    def test_unit_ellipsoid_special_case(self):
        """Orthonormal-ish design: width_i = sqrt(budget / n) exactly when Z'Z = n I."""
        n = 8
        # balanced +-1 design with orthogonal columns: Z'Z = n I
        H = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                      [1, -1, 1, -1, 1, -1, 1, -1],
                      [1, 1, -1, -1, 1, 1, -1, -1],
                      [1, -1, -1, 1, 1, -1, -1, 1]], dtype=float).T
        X = H[:, 1:]
        y = np.array([3.0, 1.0, 2.0, 5.0, 4.0, 0.0, 2.5, 1.5])
        assert np.allclose(design(X).T @ design(X), n * np.eye(4))
        p = Prism.from_data(X, y, r=0.7, units="raw")
        # budget = n * r, Ainv = I/n  => halfwidth = sqrt(n*r / n) = sqrt(r)
        for lo, hi in p.coef_bounds().values():
            assert (hi - lo) / 2 == pytest.approx(np.sqrt(0.7), rel=1e-10)


class TestDisagreementMath:
    def test_halfwidth_closed_form(self, setup):
        prism, X, y, ref = setup
        d = prism.disagreement()
        hw = ref.pred_halfwidth(X)
        np.testing.assert_allclose(d["max"] - d["optimal"], hw, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(d["optimal"] - d["min"], hw, rtol=1e-8, atol=1e-10)

    def test_extremes_attained_and_never_exceeded(self, setup):
        prism, X, y, ref = setup
        rng = np.random.default_rng(21)
        Xn = rng.standard_normal((6, X.shape[1])) * 2 + X.mean(axis=0)
        d = prism.disagreement(Xn)
        Zn = design(Xn)
        # attained: extremiser direction d = sqrt(budget / z'Ainv z) * Ainv z
        for row, z in enumerate(Zn):
            direction = ref.Ainv @ z
            scale = np.sqrt(ref.budget / (z @ direction))
            for sign, key in ((1, "max"), (-1, "min")):
                theta = ref.theta + sign * scale * direction
                assert mse(theta, X, y) == pytest.approx(ref.allowed, rel=1e-9)
                assert z @ theta == pytest.approx(d[key][row], rel=1e-8, abs=1e-9)
        # never exceeded: random members of the set stay inside [min, max]
        for theta in ref.sample_boundary(300, rng):
            pred = Zn @ theta
            assert np.all(pred <= d["max"] + 1e-8)
            assert np.all(pred >= d["min"] - 1e-8)

    def test_r_zero_has_no_disagreement(self, data):
        X, y = data
        d = Prism.from_data(X, y, r=0.0).disagreement()
        np.testing.assert_allclose(d["max"], d["min"], atol=1e-6)

    def test_extrapolation_widens_disagreement(self):
        """Leverage: predictions far from the data are less pinned down."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 2))
        y = X @ [1.0, -2.0] + rng.standard_normal(200)
        p = Prism.from_data(X, y, r=2.0)
        near = p.disagreement(np.zeros(2))
        far = p.disagreement(np.array([8.0, 8.0]))
        assert (far["max"] - far["min"])[0] > 5 * (near["max"] - near["min"])[0]

    def test_disagreement_contains_every_coef_bound_extremiser(self, setup):
        """A coefficient is a prediction at x = e_i minus x = 0: bounds must be consistent."""
        prism, X, y, ref = setup
        i = 0
        e = np.zeros(X.shape[1])
        e[i] = 1.0
        # width of slope_i is half-width of (pred(e_i) - pred(0)) in the same ellipsoid
        z = np.r_[0.0, e]
        hw = np.sqrt(ref.budget * z @ ref.Ainv @ z)
        lo, hi = prism.coef_bounds()[prism.feature_names[i]]
        assert (hi - lo) / 2 == pytest.approx(hw, rel=1e-8)


class TestDroppableMath:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_query_matches_brute_force_refit_at_every_threshold(self, seed):
        """Place the allowed loss in the gap between consecutive subset losses so the
        answer for *every* subset is unambiguous, and compare to sklearn refits."""
        X, y = make_data(seed, p=4)
        losses = subset_losses(X, y)
        ordered = np.sort(list(losses.values()))
        base = Prism.from_data(X, y).baseline_loss
        names = list(Prism.from_data(X, y).feature_names)
        for k in range(0, len(ordered) - 1, 2):
            thr = 0.5 * (ordered[k] + ordered[k + 1])
            p = Prism.from_data(X, y, r=thr - base, units="raw")
            for subset, loss in losses.items():
                got = p.query_droppable([names[j] for j in subset])
                assert got == (loss <= thr), (subset, loss, thr)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_droppable_enumeration_matches_brute_force(self, seed):
        X, y = make_data(seed, p=5)
        losses = subset_losses(X, y)
        ordered = np.sort(list(losses.values()))
        probe = Prism.from_data(X, y)
        base = probe.baseline_loss
        names = list(probe.feature_names)
        for q in (0.1, 0.3, 0.5, 0.8):
            k = int(q * (len(ordered) - 1))
            thr = 0.5 * (ordered[k] + ordered[k + 1])
            p = Prism.from_data(X, y, r=thr - base, units="raw")
            expected = {s for s, l in losses.items() if l <= thr}
            got = as_sets(p.droppable(max_terms=None), names)
            assert got == expected

    def test_max_terms_limits_subset_size(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=1000.0, units="percent")  # everything droppable
        for m in (1, 2, 3):
            subs = p.droppable(max_terms=m)
            assert all(1 <= len(s) <= m for s in subs)
            assert len(subs) == sum(len(list(combinations(range(X.shape[1]), k)))
                                    for k in range(1, m + 1))

    def test_huge_radius_makes_everything_droppable_including_all_features(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=1e9, units="percent")
        subs = p.droppable(max_terms=None)
        assert len(subs) == 2 ** X.shape[1] - 1
        assert p.query_droppable(list(p.feature_names))

    def test_zero_radius_drops_only_useless_features(self):
        """Exactly-zero coefficient feature (orthogonal to residual) is droppable at r=0."""
        n = 8
        H = np.array([[1, -1, 1, -1, 1, -1, 1, -1],
                      [1, 1, -1, -1, 1, 1, -1, -1],
                      [1, -1, -1, 1, 1, -1, -1, 1]], dtype=float).T
        X = H
        y = 2.0 * X[:, 0] + 1.0 * X[:, 1]       # third column has exactly zero effect
        p = Prism.from_data(X, y, r=0.0, units="raw")
        names = list(p.feature_names)
        assert p.query_droppable([names[2]])
        assert not p.query_droppable([names[0]])
        assert as_sets(p.droppable(max_terms=None), names) == {frozenset({2})}

    def test_subset_closed_downwards(self, setup):
        """If a subset is droppable, so is every subset of it."""
        prism, X, y, ref = setup
        names = list(prism.feature_names)
        found = as_sets(prism.droppable(max_terms=None), names)
        for s in found:
            for k in range(1, len(s)):
                for sub in combinations(sorted(s), k):
                    assert frozenset(sub) in found

    def test_loss_monotone_in_dropped_set(self, data):
        X, y = data
        losses = subset_losses(X, y)
        for s, l in losses.items():
            for t, m in losses.items():
                if s < t:
                    assert l <= m + 1e-12

    def test_droppable_grows_with_radius(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=0.1)
        names = list(p.feature_names)
        prev = set()
        for r in (0.1, 1, 5, 20, 100):
            p.update_r(r)
            cur = as_sets(p.droppable(max_terms=None), names)
            assert prev <= cur
            prev = cur

    def test_droppable_consistent_with_query_droppable(self, setup):
        prism, X, y, ref = setup
        names = list(prism.feature_names)
        found = as_sets(prism.droppable(max_terms=None), names)
        for size in (1, 2):
            for combo in combinations(range(len(names)), size):
                assert prism.query_droppable([names[j] for j in combo]) == (frozenset(combo) in found)

    def test_droppable_agrees_with_coefficient_bounds_for_single_features(self, setup):
        """Dropping one feature (refit others) is in the set iff its zero-coefficient
        profile loss <= allowed iff 0 lies inside that feature's coef bounds."""
        prism, X, y, ref = setup
        b = prism.coef_bounds()
        for i, k in enumerate(prism.feature_names):
            lo, hi = b[k]
            assert prism.query_droppable([k]) == (lo - 1e-9 <= 0.0 <= hi + 1e-9)


class TestSpecMath:
    def test_sans_features_equals_sklearn_refit(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        names = s.feature_names
        for drop in ([0], [2], [0, 3], [1, 2, 4]):
            keep = [j for j in range(X.shape[1]) if j not in drop]
            m = s.sans_features([names[j] for j in drop])
            sk = LinearRegression().fit(X[:, keep], y)
            np.testing.assert_allclose(m.coef_[drop], 0.0, atol=0.0)
            np.testing.assert_allclose(m.coef_[keep], sk.coef_, rtol=1e-7, atol=1e-9)
            assert m.intercept_ == pytest.approx(sk.intercept_, rel=1e-7, abs=1e-9)

    def test_sans_all_features_is_mean_model(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        m = s.sans_features(list(s.feature_names))
        np.testing.assert_array_equal(m.coef_, 0.0)
        assert m.intercept_ == pytest.approx(y.mean())
        assert s.mse(m) == pytest.approx(np.var(y))

    def test_from_coefs_pins_values_and_satisfies_normal_equations(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        names = s.feature_names
        target = {names[0]: 0.37, names[3]: -1.25}
        m = s.from_coefs(target)
        assert m.coef_[0] == 0.37 and m.coef_[3] == -1.25
        free = [1, 2, 4]
        resid = y - m.predict(X)
        # least-squares optimality for the free parameters
        assert np.sum(resid) == pytest.approx(0.0, abs=1e-6)
        np.testing.assert_allclose(X[:, free].T @ resid, 0.0, atol=1e-6)

    def test_from_coefs_is_profile_minimum(self, setup):
        """Perturbing any free parameter can only increase the loss."""
        prism, X, y, ref = setup
        s = prism.spec()
        m = s.from_coefs({s.feature_names[1]: 0.9})
        theta = np.r_[m.intercept_, m.coef_]
        base = mse(theta, X, y)
        rng = np.random.default_rng(4)
        free = np.array([0, 1, 3, 4, 5])  # intercept + slopes other than feature 1
        for _ in range(50):
            t = theta.copy()
            t[free] += 1e-2 * rng.standard_normal(len(free))
            assert mse(t, X, y) >= base - 1e-12

    def test_from_coefs_at_ols_value_returns_ols(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        k = s.feature_names[2]
        m = s.from_coefs({k: float(prism.coef_[2])})
        np.testing.assert_allclose(m.coef_, prism.coef_, rtol=1e-6, atol=1e-8)
        assert m.intercept_ == pytest.approx(prism.intercept_, rel=1e-6, abs=1e-8)

    def test_from_coefs_all_features_pinned_fits_intercept_only(self, data):
        X, y = data
        s = Prism.from_data(X, y).spec()
        vals = {n: float(v) for n, v in zip(s.feature_names, np.arange(X.shape[1]) * 0.1)}
        m = s.from_coefs(vals)
        expect_intercept = np.mean(y - X @ np.array(list(vals.values())))
        assert m.intercept_ == pytest.approx(expect_intercept, rel=1e-9)

    def test_from_coefs_loss_matches_ellipsoid_profile(self, setup):
        """Profiling out all other parameters gives the closed form
        MSE = MSE_ols + (c - c_ols)^2 / (n * Ainv_jj)."""
        prism, X, y, ref = setup
        s = prism.spec()
        for i in range(X.shape[1]):
            c = prism.coef_[i] + 0.3
            m = s.from_coefs({s.feature_names[i]: c})
            expected = ref.min_loss + (c - ref.theta[i + 1]) ** 2 / (ref.n * ref.Ainv[i + 1, i + 1])
            assert s.mse(m) == pytest.approx(expected, rel=1e-8)

    def test_two_pinned_coefs_match_joint_schur_complement(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        i, j = 0, 3
        c = np.array([prism.coef_[i] + 0.2, prism.coef_[j] - 0.4])
        m = s.from_coefs({s.feature_names[i]: c[0], s.feature_names[j]: c[1]})
        idx = [i + 1, j + 1]
        d = c - ref.theta[idx]
        S = ref.Ainv[np.ix_(idx, idx)]
        expected = ref.min_loss + d @ np.linalg.solve(S, d) / ref.n
        assert s.mse(m) == pytest.approx(expected, rel=1e-8)


class TestPrismSpecIntegration:
    """Spec builds concrete members that sit exactly where Prism says they should."""

    def test_spec_at_coef_bound_lies_on_boundary(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        for i, k in enumerate(prism.feature_names):
            lo, hi = prism.coef_bounds()[k]
            for bound in (lo, hi):
                m = s.from_coefs({k: bound})
                assert s.mse(m) == pytest.approx(prism.allowed_loss, rel=1e-8)

    def test_spec_inside_bounds_is_in_set_outside_is_not(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        for i, k in enumerate(prism.feature_names):
            lo, hi = prism.coef_bounds()[k]
            width = hi - lo
            assert s.mse(s.from_coefs({k: hi - 0.01 * width})) < prism.allowed_loss
            assert s.mse(s.from_coefs({k: lo + 0.01 * width})) < prism.allowed_loss
            assert s.mse(s.from_coefs({k: hi + 0.01 * width})) > prism.allowed_loss
            assert s.mse(s.from_coefs({k: lo - 0.01 * width})) > prism.allowed_loss

    def test_sans_features_loss_agrees_with_query_droppable(self, setup):
        prism, X, y, ref = setup
        s = prism.spec()
        names = list(prism.feature_names)
        for size in (1, 2, 3):
            for combo in combinations(range(len(names)), size):
                feats = [names[j] for j in combo]
                inside = s.mse(s.sans_features(feats)) <= prism.allowed_loss
                assert inside == prism.query_droppable(feats)

    def test_prediction_bounds_attained_by_spec_models_for_single_coefficient(self, setup):
        """Model built at a coef bound moves the prediction at e_i by exactly the bound."""
        prism, X, y, ref = setup
        s = prism.spec()
        k = prism.feature_names[1]
        lo, hi = prism.coef_bounds()[k]
        m = s.from_coefs({k: hi})
        np.testing.assert_allclose(m.coef_[1], hi, rtol=1e-12)


# --------------------------------------------------------------------------- #
# 3. Invariances and properties
# --------------------------------------------------------------------------- #
class TestInvariances:
    def test_row_permutation_invariance(self, data):
        X, y = data
        perm = np.random.default_rng(0).permutation(len(y))
        a = Prism.from_data(X, y, r=2.0)
        b = Prism.from_data(X[perm], y[perm], r=2.0)
        for k in a.feature_names:
            np.testing.assert_allclose(a.coef_bounds()[k], b.coef_bounds()[k], rtol=1e-8)
        names = list(a.feature_names)
        assert as_sets(a.droppable(None), names) == as_sets(b.droppable(None), names)

    def test_feature_rescaling(self, data):
        """Scaling column j by c scales its coef and bounds by 1/c; predictions,
        loss, and droppability are unchanged."""
        X, y = data
        c = np.array([1.0, 10.0, 0.1, 3.0, 1.0])
        a = Prism.from_data(X, y, r=2.0)
        b = Prism.from_data(X * c, y, r=2.0)
        for i, k in enumerate(a.feature_names):
            ka = list(a.feature_names)[i]
            kb = list(b.feature_names)[i]
            np.testing.assert_allclose(
                np.array(b.coef_bounds()[kb]), np.array(a.coef_bounds()[ka]) / c[i], rtol=1e-7
            )
        da, db = a.disagreement(), b.disagreement()
        for key in da:
            np.testing.assert_allclose(da[key], db[key], rtol=1e-7, atol=1e-8)
        assert as_sets(a.droppable(None), list(a.feature_names)) == as_sets(
            b.droppable(None), list(b.feature_names)
        )
        assert a.allowed_loss == pytest.approx(b.allowed_loss, rel=1e-9)

    def test_response_shift_only_moves_intercept(self, data):
        X, y = data
        a = Prism.from_data(X, y, r=2.0)
        b = Prism.from_data(X, y + 1000.0, r=2.0)
        np.testing.assert_allclose(a.coef_, b.coef_, rtol=1e-7, atol=1e-8)
        assert b.intercept_ == pytest.approx(a.intercept_ + 1000.0, rel=1e-9)
        for k in a.feature_names:
            np.testing.assert_allclose(a.coef_bounds()[k], b.coef_bounds()[k], rtol=1e-6, atol=1e-7)

    def test_response_scaling_scales_raw_widths_by_sqrt_budget(self, data):
        """y -> a*y: OLS coefs scale by a; in percent units budget scales by a^2,
        so half-widths scale by a as well."""
        X, y = data
        a = 4.0
        p1 = Prism.from_data(X, y, r=3.0, units="percent")
        p2 = Prism.from_data(X, a * y, r=3.0, units="percent")
        for k in p1.feature_names:
            w1 = np.ptp(p1.coef_bounds()[k])
            w2 = np.ptp(p2.coef_bounds()[k])
            assert w2 / w1 == pytest.approx(a, rel=1e-7)

    def test_duplicated_rows_do_not_change_set(self, data):
        """Stacking the data on itself leaves MSE and the ellipsoid unchanged:
        budget (n * slack) and Z'Z both double, so the bounds are identical."""
        X, y = data
        a = Prism.from_data(X, y, r=2.0)
        b = Prism.from_data(np.vstack([X, X]), np.r_[y, y], r=2.0)
        assert a.allowed_loss == pytest.approx(b.allowed_loss, rel=1e-9)
        for k in a.feature_names:
            np.testing.assert_allclose(a.coef_bounds()[k], b.coef_bounds()[k], rtol=1e-7)


# --------------------------------------------------------------------------- #
# 4. Input validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_row_mismatch(self, data):
        X, y = data
        with pytest.raises(ValueError, match="rows"):
            Prism(LinearRegression().fit(X, y), X, y[:-1])

    def test_requires_intercept(self, data):
        X, y = data
        with pytest.raises(ValueError, match="fit_intercept"):
            Prism(LinearRegression(fit_intercept=False).fit(X, y), X, y)

    def test_unfitted_model(self, data):
        X, y = data
        with pytest.raises(Exception):
            Prism(LinearRegression(), X, y)

    def test_wrong_feature_count_for_model(self, data):
        X, y = data
        m = LinearRegression().fit(X, y)
        with pytest.raises(Exception):
            Prism(m, X[:, :-1], y)

    @pytest.mark.parametrize("units", ["pct", "", "RAW"])
    def test_bad_units(self, data, units):
        X, y = data
        with pytest.raises(ValueError, match="units"):
            Prism.from_data(X, y, units=units)
        with pytest.raises(ValueError, match="units"):
            Prism.from_data(X, y).update_r(1.0, units=units)

    def test_negative_radius(self, data):
        X, y = data
        with pytest.raises(ValueError, match="non-negative"):
            Prism.from_data(X, y, r=-0.1)
        p = Prism.from_data(X, y)
        with pytest.raises(ValueError, match="non-negative"):
            p.update_r(-1.0)

    def test_failed_update_leaves_state_unchanged(self, data):
        X, y = data
        p = Prism.from_data(X, y, r=2.0, units="percent")
        before = (p.r, p.units, p.allowed_loss)
        with pytest.raises(ValueError):
            p.update_r(-1.0)
        with pytest.raises(ValueError):
            p.update_r(1.0, units="bogus")
        assert (p.r, p.units, p.allowed_loss) == before

    def test_bad_mode(self, data):
        X, y = data
        with pytest.raises(ValueError, match="mode"):
            Prism.from_data(X, y).coef_bounds(mode="median")

    def test_unknown_feature_names(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        for call in (
            lambda: p.coef_bounds(features=["nope"]),
            lambda: p.query_droppable(["nope"]),
            lambda: p.spec().sans_features("nope"),
            lambda: p.spec().from_coefs({"nope": 1.0}),
        ):
            with pytest.raises((ValueError, KeyError)):
                call()

    def test_disagreement_wrong_width(self, data):
        X, y = data
        p = Prism.from_data(X, y)
        with pytest.raises(ValueError, match="columns"):
            p.disagreement(X[:, :-1])
        with pytest.raises(ValueError, match="columns"):
            p.disagreement(np.zeros(X.shape[1] + 1))

    def test_empty_droppable_when_nothing_is_droppable(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 3))
        y = X @ [5.0, -4.0, 3.0] + 0.1 * rng.standard_normal(200)
        p = Prism.from_data(X, y, r=0.0, units="raw")
        assert p.droppable(max_terms=None) == []
        assert not p.query_droppable(list(p.feature_names))
