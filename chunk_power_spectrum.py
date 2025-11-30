import numpy as np
from numpy.fft import fft, fftfreq

def _separate_chunks(signal, chunk_size = 1825):
    """
    Separating signal into chunks.
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
        
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
        
    Returns
    -------
    chunks : list of ndarray
        List containing separated chunks of the input signal.
    """
    chunk_length = chunk_size
    num_chunks   = len(signal) // chunk_length
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_length
        end   = start + chunk_length
        chunks.append(signal[start:end])
    return chunks

def _calc_power_spectrum(signal, chunk_size = 1825):
    """
    Calculating power spectrum
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
        
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
        
    Returns
    -------
    freq : ndarray
        Frequencies corresponding to the power spectrum.
    
    sp : ndarray
        Power spectrum of the input signal.
    """
    size = np.floor(chunk_size / 2).astype(int)
    ck   = fft(signal)
    freq = fftfreq(len(signal))
    sp   = 2 * ck * ck.conj() / len(signal)**2
    return freq[:size], sp[:size]

def power_spectrum(signal, chunk_size = 1825):
    """
    Calculating averaged power spectrum over chunks.
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
        
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
        
    Returns
    -------
    freq : ndarray
        Frequencies corresponding to the power spectrum.
    
    all_sp : ndarray
        All power spectrum for each chunk of the input signal.
    """
    if len(signal) < chunk_size:
        raise ValueError("Signal length must be at least as large as chunk_size.")
    
    chunks = _separate_chunks(signal, chunk_size)
    all_sp = []
    
    for chunk in chunks:
        freq, sp = _calc_power_spectrum(chunk, chunk_size)
        sp      /= np.sum(sp) # Normalizing
        all_sp.append(sp)
    all_sp = np.array(all_sp)
    
    return freq, all_sp