import pytest
import numpy as np
from pyRedNoise.RedNoiseAnalysis import RedNoiseAnalysis
from .fixture import rmm_data

def test_RedNoiseAnalysis():
    # testing data
    x = np.linspace(0, 2 * np.pi, 10000)
    y = 2 * np.sin(x) + 0.2 * np.random.rand(10000)
    
    rna = RedNoiseAnalysis(y, lag = 1, chunk_size = 2500, n_red_noise_ens = 100, red_noise_simulate_length = 365000)
    rna.get_power_spectrum()
    
    # test if attributes are correctly assigned
    assert rna.signal.shape    == (10000,)
    assert rna.sp.shape        == (4, 1250)
    assert rna.freq.shape      == (1250,)
    assert rna.freq_theo.shape == (1250,)
    assert rna.sp_theo.shape   == (1250,)
    assert rna.sp_red.shape    == (100, 4, 1250)
    assert rna.freq_red.shape  == (1250,)

def test_plotting(rmm_data):
    rmm1 = rmm_data[0]
    rna  = RedNoiseAnalysis(rmm1, lag = 1, chunk_size = 1825, n_red_noise_ens = 100, red_noise_simulate_length = 365000)
    rna.get_power_spectrum()
    # test plotting functions
    try:
        rna.plot_power_spectrum()
    except Exception as e:
        pytest.fail(f"Plotting methods raised an exception: {e}")