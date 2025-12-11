import pytest
import numpy as np
from numpy.fft import fft, fftfreq
from pyRedNoise.chunk_power_spectrum import _separate_chunks, _calc_power_spectrum, power_spectrum

def test_separate_chunks():
    
    # test if ValueError is raised for too short signal
    short_signal = np.random.rand(1000)
    with pytest.raises(ValueError):
        _separate_chunks(short_signal, chunk_size=1825)
    
    # test normal case
    signal = np.random.rand(5000)
    chunks = _separate_chunks(signal, chunk_size=1825)
    assert chunks.shape == (2, 1825)
    
def test_calc_power_spectrum():
    
    # testing data
    x = np.linspace(0, 2 * np.pi, 5000)
    y = 2 * np.sin(x) + 0.2 * np.random.rand(5000)
    
    # test the output size
    freq, sp = _calc_power_spectrum(y)
    assert len(freq) == 2500
    assert len(sp)   == 2500
    
    # test the power spectrum values
    assert np.amax(sp) == pytest.approx(2**2/2, rel = 1e-2)
    
def test_power_spectrum():
    # testing data
    x = np.linspace(0, 2 * np.pi, 5000)
    y = 2 * np.sin(x) + 0.2 * np.random.rand(5000)
    
    freq, sp = power_spectrum(y, chunk_size = 1000)
    
    # test the output size
    assert freq.shape == (500,)
    assert sp.shape   == (5, 500,)