import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

class Spectrum:
    def __init__(self, frequency, wavelength, flux, noise):

        self.frequency = np.array(frequency)
        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        self.noise = np.array(noise)

    def spectrum_region_from_boundary(self,i_start, i_end):
      return Spectrum(self.frequency[i_start:i_end], self.wavelength[i_start:i_end], 
                      self.flux[i_start:i_end], self.noise[i_start:i_end])

    def estimate_n(self):
        n = int(argrelextrema(gaussian_filter(self.flux, 3), np.less)[0].shape[0])
        if n < 4:
            n = 1
            return n