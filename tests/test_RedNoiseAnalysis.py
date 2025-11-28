import numpy as np
import pytest

from RedNoiseAnalysis import RedNoiseAnalysis


def test_rednoiseanalysis_init_sets_attributes():
    """Basic sanity check on initialization and default attribute values."""
    signal = np.arange(100.0)
    rna = RedNoiseAnalysis(signal, lag=2, chunk_size=20, red_noise_simulate_length=500)

    # Public attributes
    assert np.array_equal(rna.signal, signal)
    assert rna.sp is None
    assert rna.freq is None
    assert rna.freq_theo is None
    assert rna.sp_theo is None
    assert rna.sp_red is None
    assert rna.freq_red is None

    # Internal attributes
    assert rna._lag == 2
    assert rna._chunk_size == 20
    assert rna._simulate_length == 500
    assert rna._a is None
    assert rna._red_noise is None


def test_rednoiseanalysis_get_power_spectrum_populates_fields():
    """
    get_power_spectrum() should compute:
      - power spectrum of the original signal
      - theoretical red noise spectrum
      - simulated red noise and its spectrum
    and populate all corresponding attributes.
    """
    # Use a moderate length and small simulate_length to keep test fast
    rng = np.random.default_rng(0)
    signal = rng.normal(size=1000)
    lag = 1
    chunk_size = 50
    simulate_length = 5000

    rna = RedNoiseAnalysis(
        signal,
        lag=lag,
        chunk_size=chunk_size,
        red_noise_simulate_length=simulate_length,
    )

    rna.get_power_spectrum()

    # Original signal spectrum
    assert rna.freq is not None
    assert rna.sp is not None
    assert rna.freq.shape == rna.sp.shape

    # Theoretical red noise spectrum
    assert rna.freq_theo is not None
    assert rna.sp_theo is not None
    assert rna.freq_theo.shape == rna.sp_theo.shape

    # Simulated red noise and its spectrum
    assert rna._red_noise is not None
    assert rna._red_noise.shape == (simulate_length,)
    assert rna.freq_red is not None
    assert rna.sp_red is not None
    assert rna.freq_red.shape == rna.sp_red.shape

    # Internal AR(1) coefficient should be a scalar
    assert isinstance(rna._a, (float, np.floating))


def test_rednoiseanalysis_red_noise_matches_a_lag1_acf():
    """
    The simulated red noise stored in _red_noise should behave like an AR(1)
    with coefficient approximately equal to rna._a: lag-1 autocorrelation ≈ a.
    """
    rng = np.random.default_rng(1)
    signal = rng.normal(size=2000)
    lag = 1
    chunk_size = 100
    simulate_length = 20_000  # a bit larger for decent statistics

    rna = RedNoiseAnalysis(
        signal,
        lag=lag,
        chunk_size=chunk_size,
        red_noise_simulate_length=simulate_length,
    )
    rna.get_power_spectrum()

    # Lag-1 autocorrelation of simulated red noise
    red = rna._red_noise
    acf_lag1 = np.corrcoef(red[:-1], red[1:])[0, 1]

    # rna._a is the estimated AR(1) coefficient from the original signal
    a_est = rna._a

    # They should be roughly consistent
    assert acf_lag1 == pytest.approx(a_est, rel=0.2)


def test_rednoiseanalysis_power_spectrum_consistency_between_original_and_red():
    """
    Check that the frequency grids for original and red-noise spectra have the same length
    (since they use the same chunk_size).
    """
    rng = np.random.default_rng(2)
    signal = rng.normal(size=1500)
    lag = 1
    chunk_size = 60
    simulate_length = 6000

    rna = RedNoiseAnalysis(
        signal,
        lag=lag,
        chunk_size=chunk_size,
        red_noise_simulate_length=simulate_length,
    )
    rna.get_power_spectrum()

    assert rna.freq is not None
    assert rna.freq_red is not None
    assert rna.freq.shape == rna.freq_red.shape
