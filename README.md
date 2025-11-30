```
class RedNoiseAnalysis(
    signal,
    lag = 1,
    chunk_size = 1825,
    n_red_noise_ens = 1000,
    red_noise_simulate_length = 365000
)
```
This is a package that performing temporal power spectrum analysis, and comparing given signal with theoretical red noise and simulated red noise.

## Usage
Calculating power spectrum of given signal, and red noise which has similar memory with given signal.
```
from pyRedNoise.RedNoiseAnalysis import RedNoiseAnalysis
signal = np.load("signal.npy")

rna = RedNoiseAnalysis(signal)
rna.get_power_spectrum()

# Given signal
rna.signal

# power spectrum of given signal
rna.sp

# frequencies of given signal
rna.freq

# theoretical red noise power spectrum
rna.sp_theo

# theoretical red noise frequencies
rna.freq_theo

# simulated red noise power spectrum
rna.sp_red

# simulated red noise frequencies
rna.freq_red
```

Fast visualization
```
rna.plot_power_spectrum() # show and save the plot at the same time
```

## Notes
Details of derivation, theory, and implementation can be found in https://github.com/Prcpltwfkwd-TW/Power_Spectrum_Analysis.git
