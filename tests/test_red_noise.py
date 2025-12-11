import pytest
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.linalg import logm, expm
from pyRedNoise.red_noise import calc_a

def test_calc_a():
    
    # testing data
    a = np.array([0.95 ** i for i in range(100)])
    
    # lag 1
    a1 = calc_a(a, lag = 1)
    assert a1 == pytest.approx(0.95, rel = 1e-2)
    
    # lag 5
    a5 = calc_a(a, lag = 5)
    assert a5 == pytest.approx(0.95**5, rel = 1e-2)