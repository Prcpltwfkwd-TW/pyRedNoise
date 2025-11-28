from chunk_power_spectrum import power_spectrum
from red_noise import create_red_noise, theoretical_red_noise_power_spectrum, calc_a

class RedNoiseAnalysis:
    """
    Performing red noise analysis on a given signal.
    
    Parameters
    ----------
    signal : array_like
        Input signal (1D array).
    
    lag : int, optional
        Lag at which to calculate the regression coefficient. Default is 1.
        
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
        
    red_noise_simulate_length : int, optional
        Length of the simulated red noise signal. Default is 365000, 1000 years for daily data.
    
    Public Attributes
    -----------------
    signal : array_like
        Input signal (1D array).
    
    sp : ndarray
        Power spectrum of the input signal.
    
    freq : ndarray
        Frequencies corresponding to the power spectrum of the input signal.
    
    freq_theo : ndarray
        Frequencies corresponding to the theoretical red noise power spectrum.
    
    sp_theo : ndarray
        Theoretical red noise power spectrum calculated according to the input signal.
        
    sp_red : ndarray
        Power spectrum of the simulated red noise.
    
    freq_red : ndarray
        Frequencies corresponding to the power spectrum of the simulated red noise.
    ----------
    """
    def __init__(self, signal, lag = 1, chunk_size = 1825, red_noise_simulate_length = 365000):
        self.signal           = signal
        self._lag             = lag
        self._chunk_size      = chunk_size
        self._simulate_length = red_noise_simulate_length
        
        self.sp   = None
        self.freq = None
        
        self._a         = None
        self.freq_theo  = None
        self.sp_theo    = None
        self._red_noise = None
        self.sp_red     = None
        self.freq_red   = None
        
    def get_power_spectrum(self):
        """
        Calculating power spectrum of the input signal and corresponding red noise.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        self.freq, self.sp = power_spectrum(self.signal, self._chunk_size)
        
        self._a                      = calc_a(self.signal, self._lag)
        self.freq_theo, self.sp_theo = theoretical_red_noise_power_spectrum(self.signal, self._lag, self._chunk_size)
        self._red_noise              = create_red_noise(self._a, self._simulate_length)
        self.freq_red, self.sp_red   = power_spectrum(self._red_noise, self._chunk_size)