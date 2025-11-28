import random
import numpy as np
from numpy.fft import fft, fftfreq
from scipy.linalg import logm, expm

from chunk_power_spectrum import power_spectrum

# lead-lag regression coefficient
def calc_a(signal, lag = 1):
    """
    Calculating the lead-lag regression coefficient at specified lag.
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
    
    lag : int, optional
        Lag at which to calculate the regression coefficient. Default is 1.
        Note that the lag can be greater than 1.
        
    Returns
    -------
    a : float
        Lead-lag regression coefficient at the specified lag.
        If the given lag is greater than 1, the lag-th root of the regression coefficient matrix is returned.
    """
    x0 = signal[:-lag]
    x1 = signal[lag:]
    x0 = np.reshape(x0, (1, len(x0)))
    x1 = np.reshape(x1, (1, len(x1)))
    c0 = np.matmul(x0, x0.T)
    ct = np.matmul(x1, x0.T)
    a  = np.matmul(ct, np.linalg.inv(c0))
    a  = np.squeeze(expm(logm(a) / lag)) # take the lag-th root to get the regression coefficient at lag 1
    return a

# red noise model
def _time_integral(x: float, a: float, e: float):
    """
    Time integration of red noise model.
    
    Parameters
    ----------
    x : float
        Current value of the signal.
    
    a : float
        Lead-lag regression coefficient.
    
    e : float
        Noise term.
        
    Returns
    -------
    x_new : float
        New value of the signal after time integration.
    """
    return np.dot(a, x) + e

def _calc_noise(a, size = 1000):
    """
    Calculating noise term for red noise model.
    
    Parameters
    ----------
    a : float
        Lead-lag regression coefficient.
    
    size : int, optional
        Size of the white noise to be generated. Default is 1000.
        
    Returns
    -------
    noise : float
        Noise term for the red noise model.
    """
    sigma       = np.sqrt(1 - a**2)
    white_noise = np.random.normal(0, sigma, size = size)
    noise       = np.random.choice(white_noise)
    return noise

# Creating red noise, both theoretical and simulated
def theoretical_red_noise_power_spectrum(signal, lag = 1, chunk_size = 1825):
    """
    Creating theoretical red noise power spectrum according to given signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
        
    lag : int, optional
        Lag at which to calculate the regression coefficient. Default is 1.
    
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
    
    Returns
    -------
    freq : ndarray
        Frequencies corresponding to the power spectrum.
    
    sp : ndarray
        Theoretical red noise power spectrum of the input signal.
    """
    a    = calc_a(signal, lag)
    size = np.floor(chunk_size / 2).astype(int)
    freq = fftfreq(chunk_size)
    sp   = (1 - a**2) / (1 + a**2 - 2 * a * np.cos(2 * np.pi * freq))
    return freq[:size], sp.real[:size]

def create_red_noise(a, simulate_length = 365000):
    """
    Creating simulated red noise according to given lead-lag regression coefficient.
    
    Parameters
    ----------
    a : float
        Lead-lag regression coefficient.
        
    simulate_length : int, optional
        Length of the simulated red noise signal. Default is 365000, 1000 years for daily data.
        
    Returns
    -------
    red : ndarray
        Simulated red noise signal.
    """
    red    = np.zeros(simulate_length)
    red[0] = np.random.normal(0, 1)
    for i in range(simulate_length - 1):
        noise    = _calc_noise(a)
        red[i+1] = _time_integral(red[i], a, noise)
    return red