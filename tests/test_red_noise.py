import numpy as np
import pytest

import pyRedNoise
from pyRedNoise.red_noise import calc_a, _time_integral, _calc_noise, create_red_noise

# Tests for calc_a function
def test_calc_a_lag1_ar1_deterministic():
    """
    For a deterministic AR(1) sequence x_n = rho^n with no noise,
    the regression coefficient at lag=1 should be exactly rho.
    """
    rho = 0.8
    N = 200  # reasonably long to avoid tiny-length effects

    n = np.arange(N)
    signal = rho ** n

    a = calc_a(signal, lag=1)

    assert a == pytest.approx(rho, rel=1e-6, abs=1e-8)


def test_calc_a_lagK_returns_lag1_coefficient():
    """
    For the same deterministic AR(1) sequence x_n = rho^n,
    the regression coefficient at lag=K is rho^K, but the function
    takes the K-th root via logm/expm and should return rho again.
    """
    rho = 0.6
    N = 300
    K = 5  # lag > 1

    n = np.arange(N)
    signal = rho ** n

    a = calc_a(signal, lag=K)

    # Should recover the underlying lag-1 coefficient rho
    assert a == pytest.approx(rho, rel=1e-6, abs=1e-8)


def test_calc_a_constant_signal():
    """
    For a constant signal, x(t+lag) = x(t),
    so the regression coefficient should be 1 for any valid lag.
    """
    N = 100
    signal = np.ones(N)
    lag = 10

    a = calc_a(signal, lag=lag)

    assert a == pytest.approx(1.0, rel=1e-6, abs=1e-8)


def test_calc_a_sensitivity_to_lag_sign():
    """
    Ensure that different lags generally give the same underlying lag-1
    coefficient for an AR(1)-like process, even with small noise.
    """
    rng = np.random.default_rng(0)
    rho = 0.7
    N = 500

    # Noisy AR(1) realization
    x = np.zeros(N)
    noise = rng.normal(scale=0.1, size=N)
    for t in range(1, N):
        x[t] = rho * x[t - 1] + noise[t]

    a1 = calc_a(x, lag=1)
    a3 = calc_a(x, lag=3)

    # Both should be close to the true rho, and close to each other
    assert a1 == pytest.approx(rho, rel=0.1)   # allow some noise-induced error
    assert a3 == pytest.approx(rho, rel=0.1)
    assert a1 == pytest.approx(a3, rel=0.1)

# Tests for red noise model
def test_time_integral_basic():
    """Check that the AR(1) update rule x_new = a*x + e behaves correctly."""
    x = 2.0
    a = 0.5
    e = -1.0
    assert _time_integral(x, a, e) == pytest.approx(0.5 * 2.0 - 1.0)


def test_time_integral_zero_a():
    """If a=0, the update should ignore x and return e."""
    x = 5.0
    a = 0.0
    e = -2.0
    assert _time_integral(x, a, e) == pytest.approx(e)

def test_calc_noise_variance():
    """
    Noise variance = 1 - a^2 by model design.
    Generate many samples and test empirical variance.
    """
    a = 0.8
    N = 50_000

    noises = np.array([_calc_noise(a) for _ in range(N)])
    empirical_var = noises.var()
    expected_var = 1 - a**2

    assert empirical_var == pytest.approx(expected_var, rel=0.1)


def test_calc_noise_zero_a():
    """
    When a = 0, noise variance must be 1.
    """
    a = 0.0
    noises = np.array([_calc_noise(a) for _ in range(50000)])
    assert noises.var() == pytest.approx(1.0, rel=0.1)


def test_create_red_noise_length():
    """Simulated red noise should have the requested length."""
    a = 0.4
    L = 10000
    red = create_red_noise(a, simulate_length=L)
    assert len(red) == L


def test_create_red_noise_ar1_autocorrelation():
    """
    Red noise is an AR(1) process: lag-1 autocorrelation ≈ a.
    """
    a = 0.7
    L = 200_000
    red = create_red_noise(a, simulate_length=L)

    # sample lag-1 autocorrelation
    r = np.corrcoef(red[:-1], red[1:])[0, 1]

    assert r == pytest.approx(a, rel=0.05)


def test_create_red_noise_unit_variance():
    """
    Stationary AR(1) with noise variance (1 - a^2)
    → total variance should be 1.
    """
    a = 0.9
    L = 200_000

    red = create_red_noise(a, simulate_length=L)
    variance = red.var()

    assert variance == pytest.approx(1.0, rel=0.05)


def test_create_red_noise_white_noise_case():
    """
    If a=0, the process is pure white noise with variance 1.
    """
    a = 0.0
    L = 100_000
    red = create_red_noise(a, simulate_length=L)
    assert red.var() == pytest.approx(1.0, rel=0.05)
    # correlation should be ~ 0
    r = np.corrcoef(red[:-1], red[1:])[0, 1]
    assert abs(r) < 0.02


def test_create_red_noise_high_a_stability():
    """
    Check stability and correct scaling for high persistence (a=0.99).
    """
    a = 0.99
    L = 200_000
    red = create_red_noise(a, simulate_length=L)

    # Variance should still be ~1
    assert red.var() == pytest.approx(1.0, rel=0.1)

    # Lag-1 AC ≈ 0.99
    r = np.corrcoef(red[:-1], red[1:])[0, 1]
    assert r == pytest.approx(a, rel=0.05)
