import random
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from .chunk_power_spectrum import power_spectrum
from .red_noise import create_red_noise, theoretical_red_noise_power_spectrum, calc_a

class RedNoiseAnalysis:
    """
    Performing red noise analysis on a given signal.
    
    Parameters
    ----------
    signal : array_like (time,)
        Input signal (1D array).
    
    lag : int, optional
        Lag at which to calculate the regression coefficient. Default is 1.
        
    chunk_size : int, optional
        Chunk size. Default is 1825, 5 years for daily data.
        
    red_noise_simulate_length : int, optional
        Length of the simulated red noise signal. Default is 365000, 1000 years for daily data.
    
    Public Attributes
    -----------------
    signal : array_like (time,)
        Input signal (1D array).
    
    sp : ndarray (n_chunks, size)
        Power spectrum of the input signal.
    
    freq : ndarray (size,)
        Frequencies corresponding to the power spectrum of the input signal.
    
    freq_theo : ndarray (size,)
        Frequencies corresponding to the theoretical red noise power spectrum.
    
    sp_theo : ndarray (size,)
        Theoretical red noise power spectrum calculated according to the input signal.
        
    sp_red : ndarray (n_red_noise_ens, n_chunks, size)
        Power spectrum of the simulated red noise.
    
    freq_red : ndarray (size,)
        Frequencies corresponding to the power spectrum of the simulated red noise.
    """
    def __init__(self, signal, lag = 1, chunk_size = 1825, n_red_noise_ens = 1000, red_noise_simulate_length = 365000):
        self.signal           = signal
        self._lag             = lag
        self._chunk_size      = chunk_size
        self._n_red_ens       = n_red_noise_ens
        self._simulate_length = red_noise_simulate_length
        
        self.sp   = None
        self.freq = None
        
        self._a         = None
        self.freq_theo  = None
        self.sp_theo    = None
        self._red_noise = None
        self.sp_red     = None
        self.freq_red   = None
    
    def _create_red_noise_sample(self):
        """
        Bootstrapping chunks of red noise samples from the simulated red noise.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        red : ndarray (chunk_size,)
            Chunks of red noise samples.
        """
        pos  = random.choice(np.arange(self._simulate_length - len(self.signal)))
        red  = self._red_noise[pos:pos+len(self.signal)]
        red /= red.std()
        return red
        
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
        # Signal power spectrum
        print("Calculating power spectrum of the input signal...")
        self.freq, self.sp = power_spectrum(self.signal, self._chunk_size)
        
        # Theoretical red noise power spectrum
        print("Calculating theoretical red noise power spectrum...")
        self._a                      = calc_a(self.signal, self._lag)
        self.freq_theo, self.sp_theo = theoretical_red_noise_power_spectrum(self.signal, self._lag, self._chunk_size)
        
        # Simulated red noise power spectrum
        print("Calculating simulated red noise power spectrum...")
        self._red_noise = create_red_noise(self._a, self._simulate_length)
        self.sp_red     = np.zeros((self._n_red_ens,) + self.sp.shape)
        for _ in range(self._n_red_ens):
            red = self._create_red_noise_sample() # Bootstrapping
            self.freq_red, self.sp_red[_] = power_spectrum(red, self._chunk_size)
    
    def plot_power_spectrum(self, title = "Power Spectrum of Signal and Corresponding Red Noise", fig_name = "red_noise_analysis.png"):
        """
        Plotting power spectrum of the input signal and corresponding red noise.
        
        Parameters
        ----------
        fig_name : str, optional
            Name of the output figure file. Default is "red_noise_analysis.png".
        
        Returns
        -------
        None
        
        Note that get_power_spectrum() should be called before this method.
        The plot will be saved as a PNG file with the given name.
        This method is only for fast visualization. If you have specific plotting needs, calling this method is not suggested.
        """
        print("Plotting power spectrum...")
        print("The plot will be saved as a PNG file with the given name.")
        print("This method is only for fast visualization. If you have specific plotting needs, calling this method is not suggested.")
        sp_red_ens_mean  = self.sp_red.mean(axis = 1).mean(axis = 0)
        sp_red_ens_std   = self.sp_red.mean(axis = 1).std(axis = 0)
        sp_conf_ens_high = stats.norm.interval(0.95, sp_red_ens_mean, sp_red_ens_std)[1]
        sp_conf_ens_low  = stats.norm.interval(0.95, sp_red_ens_mean, sp_red_ens_std)[0]
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.freq, self.sp.mean(axis = 0) / np.sum(self.sp.mean(axis = 0)), color = "k", linewidth = 1, label = "Signal")
        plt.plot(self.freq_red, sp_red_ens_mean / np.sum(sp_red_ens_mean), color = "dimgrey", linestyle = "--", label = "Simulated Red Noise")
        plt.fill_between(self.freq_red, sp_conf_ens_low / np.sum(sp_red_ens_mean), sp_conf_ens_high / np.sum(sp_red_ens_mean), color = "grey", alpha = 0.5)
        plt.plot(self.freq_theo, self.sp_theo / np.sum(self.sp_theo), color = "k", linestyle = ":", label = "Theoretical Red Noise")
        plt.legend(fontsize = 14)
        plt.xscale("log")
        plt.xlabel("frequency [timestep$^{-1}$]", fontsize = 14)
        plt.xticks(fontsize = 14)
        plt.ylabel("Explained Variance", fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.title(title, fontsize = 16)
        plt.savefig(fig_name, dpi = 300)
        plt.show()