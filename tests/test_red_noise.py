import pytest
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.linalg import logm, expm
from pyRedNoise.red_noise import calc_a, create_red_noise
from .fixture import rmm_data

def test_calc_a():
    
    # testing data
    a = np.array([0.95 ** i for i in range(100)])
    
    # lag 1
    a1 = calc_a(a, lag = 1)
    assert a1 == pytest.approx(0.95, rel = 1e-2)
    
    # lag 5
    a5 = calc_a(a, lag = 5)
    assert a5 == pytest.approx(0.95**5, rel = 1e-2)

def test_red_noise(rmm_data):
    rmm1   = rmm_data[0]
    a_rmm1 = calc_a(rmm1, lag = 1)
    red    = create_red_noise(a_rmm1)
    a_red  = calc_a(red, lag = 1)
    assert a_red == pytest.approx(a_rmm1, rel = 1e-2)