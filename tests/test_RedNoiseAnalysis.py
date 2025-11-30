import numpy as np
import matplotlib
import pytest

from ..red_noise import create_red_noise
from ..RedNoiseAnalysis import RedNoiseAnalysis

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
    assert rna.freq.shape[0] == rna.sp.shape[1]

    # Theoretical red noise spectrum
    assert rna.freq_theo is not None
    assert rna.sp_theo is not None
    assert rna.freq_theo.shape == rna.sp_theo.shape

    # Simulated red noise and its spectrum
    assert rna._red_noise is not None
    assert rna._red_noise.shape == (simulate_length,)
    assert rna.freq_red is not None
    assert rna.sp_red is not None
    assert rna.freq_red.shape[0] == rna.sp_red.shape[1]

    # Internal AR(1) coefficient should be an array
    assert isinstance(rna._a, (np.ndarray, np.floating))


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

# Use non-interactive backend for tests
matplotlib.use("Agg")
def test_plot_power_spectrum_creates_png(tmp_path):
    """
    plot_power_spectrum() should:
      - run without errors when all required attributes are set
      - create a non-empty PNG file at the given path
    """
    # --- Prepare a dummy instance ---
    signal = np.arange(100.0)
    rna = RedNoiseAnalysis(signal)

    # Create fake spectra with shapes consistent with the method’s expectations
    n_freq = 32
    n_ens_signal = 5
    n_ens_red_outer = 3
    n_ens_red_inner = 4

    # sp: shape (n_ens_signal, n_freq) since we do sp.mean(axis=0)
    rna.sp = np.random.rand(n_ens_signal, n_freq)
    rna.freq = np.linspace(1e-3, 0.5, n_freq)

    # sp_red: shape (n_ens_red_outer, n_ens_red_inner, n_freq)
    # because method does .mean(axis=1).mean(axis=0)
    rna.sp_red = np.random.rand(n_ens_red_outer, n_ens_red_inner, n_freq)
    rna.freq_red = np.linspace(1e-3, 0.5, n_freq)

    # theoretical spectrum: just make arrays with matching length
    rna.freq_theo = np.linspace(1e-3, 0.5, n_freq)
    rna.sp_theo = np.random.rand(n_freq)

    # --- Call the plotting method ---
    fig_path = tmp_path / "test_red_noise_plot.png"
    rna.plot_power_spectrum(fig_name=str(fig_path))

    # --- Assertions ---
    assert fig_path.exists(), "PNG file was not created by plot_power_spectrum"
    assert fig_path.stat().st_size > 0, "PNG file is empty"