from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema
import numpy as np
from copy import copy
from vpfits import VPfit
import gc

class VPregion():

    def __init__(self, frequency_array, flux_array, noise_array, voigt=False, chi_limit=1.5,):
        self.frequency_array = frequency_array
        self.flux_array =flux_array
        self.noise_array = noise_array
        self.voigt = voigt
        self.chi_limit = chi_limit
        self.num_pixels = len(flux_array)

        self.estimate_n()
        self.set_freedom()

    def estimate_n(self):
        """
        Make initial guess for number of local minima in the region.
            
        Smooth the spectra with a gaussian and find the number of local minima.
        as a safety precaucion, set the initial guess for number of profiles to 1 if
        there are less than 4 local minima.

        Args:
            flux_array (numpy array)
        """
            
        self.n = argrelextrema(gaussian_filter(self.flux_array, 3), np.less)[0].shape[0]
        if self.n < 4:
            self.n = 1

    def set_freedom(self):
        # number of degrees of freedom = number of data points + number of parameters
        self.freedom = self.num_pixels - 3*self.n


    def region_fit(self, verbose=True, iterations=3000, thin=15, burn=300):
        """
        Fit the line region with n Gaussian/Voigt profiles using a BIC method.

        Args:
            frequency_array (numpy array)
            flux_array (numpy array)
            n (int) initial guess for number of profiles
            voigt (Boolean): switch to fit as Voigt profiles or Gaussians
            chi_limit (float): upper limit for reduced chi squared
            iterations, thin, burn (int): MCMC parameters
        """
        first = True
        finished = False
        if verbose:
            print("Setting initial number of lines to: {}".format(self.n))
        while not finished:
            vpfit = VPfit()
            vpfit.find_bic(self.frequency_array, self.flux_array, self.n, self.noise_array, self.freedom, voigt=self.voigt, iterations=iterations, burn=burn)
            if first:
                first = False
                self.n += 1
                bic_old = vpfit.bic_array[-1]
                vpfit_old = copy(vpfit)
                del vpfit
            else:
                if bic_old > np.average(vpfit.bic_array):
                    if verbose:
                        print("Old BIC value of {:.2f} is greater than the current {:.2f}.".format(bic_old, np.average(vpfit.bic_array)))
                    bic_old = np.average(vpfit.bic_array)
                    vpfit_old = copy(vpfit)
                    if np.average(vpfit.red_chi_array) < self.chi_limit:
                        if verbose:
                            print("Reduced Chi squared is less than {}".format(self.chi_limit))
                            print("Final n={}".format(self.n))
                        finished = True
                        continue
                    self.n += 1
                    del vpfit
                    if verbose:
                        print("Increasing the number of lines to: {}".format(self.n))
                else:
                    if verbose:
                        print("BIC increased with increasing the line number, stopping.")
                        self.n -= 1
                        print("Final n={}.".format(self.n))
                    finished = True
                    continue
        gc.collect()
        self.fit = vpfit_old
