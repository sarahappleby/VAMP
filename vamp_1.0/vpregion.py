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


    def fit_region_pygadds(start, end, wavelength, flux, noise, voigt=False, BIC_factor=1.1,
            chi_limit=2.5, ntries=5, iterations=3000, thin=15, burn=300):
        """
        Fit a region with Gaussian/Voigt profiles using PyMC.
        Returns instance of VPfit() that contains all the fitted line information.

        Args:
            start, end: starting and ending pixel numbers for given region
            wavelength (numpy array)
            flux (numpy array)
            noise (numpy array)
            voigt (Boolean): switch to fit as Voigt profiles or Gaussians
            BIC_factor (float): factor by which BIC must be lowered in order to accept the new fit
            chi_limit (float): limit for satisfactory reduced chi squared
            ntries (int): max number of trials with a given number of lines before adding new line
            iterations, thin, burn (int): MCMC parameters
        """

        # initialise chisq loop
        chisq = np.inf
        tries = ntries
        frequency = np.flip(physics.c.in_units_of('Angstrom s**-1')/wavelength[start:end],0)  # we fit in frequency space
        n = estimate_n(flux)
        # Add lines until we get a reasonable chi-square
        while chisq > chi_limit:
            vpfit = VPfit()
            vpfit.find_bic(frequency, flux[start:end], n, noise[start:end], voigt=voigt, iterations=iterations, burn=burn)
            chisq = np.average(vpfit.red_chi_array)
            if chisq > chi_limit:   
                if tries > 0: tries -= 1  # keep trying, with same number of lines
                else: 
                    n += 1  # ok, give up and add a line
                    tries = ntries
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print "VAMP : Found %d lines, giving chisq=%g."%(n,chisq),

        # Try adding lines so long as each new line lowers chisq and also lowers the BIC by more BIC_factor
        vpfit_old = copy(vpfit)
        if BIC_factor is not None:
            vpfit.find_bic(frequency, flux[start:end], n+1, noise[start:end], voigt=voigt, iterations=iterations, burn=burn)
            while np.average(vpfit.red_chi_array) < np.average(vpfit_old.red_chi_array) and np.average(vpfit.bic_array) < BIC_factor*np.average(vpfit_old.bic_array):
                n += 1
                if environment.verbose >= environment.VERBOSE_NORMAL:
                    print " Adding line %d: BIC=%g -> %g, chisq=%g -> %g."%(n,np.average(vpfit_old.bic_array),np.average(vpfit.bic_array),np.average(vpfit_old.red_chi_array),np.average(vpfit.red_chi_array)),
                vpfit_old = copy(vpfit)  # accept the new line and try to add another
                vpfit.find_bic(frequency, flux[start:end], n+1, noise[start:end], voigt=voigt, iterations=iterations, burn=burn)

        # We have the final fit
        if environment.verbose >= environment.VERBOSE_NORMAL:
            print " Fitted %d lines: BIC=%g, chisq=%g"%(n,np.average(vpfit_old.bic_array),np.average(vpfit_old.red_chi_array))
        return vpfit_old,vpfit_old.red_chi_array[0]
