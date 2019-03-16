"""
VPfit

Fit Voigt Profiles using MCMC. Uses bayesian model selection to pick the appropriate number 
of profiles for a given absorption.

The main method for fitting absorption spectra is 'fit_spectrum'. 

The main class containing the fitting functionality is `VPfit`.

See `__init__` for example usage.

"""

import matplotlib
matplotlib.use('agg')

import random
import datetime

import numpy as np
import pandas as pd

import pymc as mc

import matplotlib.pyplot as plt

# Voigt modules
# from scipy.special import wofz
from astropy.modeling.models import Voigt1D

# peak finding modules
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter

# gaussian smoothing modules
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

from copy import copy
import gc

from physics import *

#TODO: change the way that "Models are instantiated" (need to understand that first)
import warnings
warnings.filterwarnings("ignore", message="Instantiating a Model object directly is deprecated. We recommend passing variables directly to the Model subclass.")


class VPfit():

    def __init__(self, noise=None):
        """
        Initialise noise variable
        """
        self.std_deviation = 1./(mc.Uniform("sd", 0, 1) if noise is None else noise)**2
        self.verbose = False


    @staticmethod
    def GaussFunction(x, amplitude, centroid, sigma):
        """
        Gaussian.

        Args:
            x (numpy array): wavelength array
            amplitude (float)
            centroid (float): must be between the limits of wavelength_array
            sigma (float)
        """
        return amplitude * np.exp(-0.5 * ((x - centroid) / sigma) ** 2)


    @staticmethod
    def VoigtFunction(x, centroid, amplitude, L_fwhm, G_fwhm):
        """
        Return the Voigt line shape at x with Lorentzian component HWHM gamma
        and Gaussian component HWHM alpha.

        Source: http://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/

        Args:
            x (numpy array): wavelength array
            centroid (float): center of profile
            alpha (float): Gaussian HWHM
            gamma (float): Lorentzian HWHM
        """

        #sigma = alpha / np.sqrt(2 * np.log(2))
        #return np.real(wofz((x - centroid + 1j*gamma)/sigma/np.sqrt(2))) / sigma / np.sqrt(2*np.pi)

        v = Voigt1D(x_0=centroid, amplitude_L=amplitude, fwhm_L=L_fwhm, fwhm_G=G_fwhm)
        return v(x)


    @staticmethod
    def GaussianWidth(G_fwhm):
        """
        Find the std deviation of the Gaussian associated with a Voigt profile, from the
        Full Width Half Maximum (FWHM)

        Args:
            G_fwhm (float): Full Width Half Maximum (FWHM)
        """
        return G_fwhm / (2.* np.sqrt(2.*np.log(2.)))
    

    def trunc_normal(self, name, mu, tau, a, b):
        """
        Take a sample from a truncated Gaussian distribution with PyMC Normal.
        Args:
            name (string): name of the component
            mu (float): mean of the distribution
            tau (float): precision of the distribution, which corresponds to 1/sigma**2 (tau > 0).
            a (float): lower bound of the distribution
            b (float): upper bound of the distribution
        """
        for i in range(1000):
            x = pymc.Normal(name, mu, tau)
            if (a < x.value) and (b > x.value):
                break
        if (a > x.value) or (b < x.value):
            raise ValueError('Computational time exceeded.')    
        return x

    @staticmethod
    def Chisquared(observed, expected, noise):
        """
        Calculate the chi-squared goodness of fit

        Args:
            observed (array)
            expected (array): same shape as observed
        """
        return sum( ((observed - expected) /  noise)**2 )


    @staticmethod
    def ReducedChisquared(observed, expected, noise, freedom):
        """
        Calculate the reduced chih-squared goodness of fit

        Args:
            observed (array)
            expected (array): same shape as observed
            freedom (int): degrees of freedom
        """
        return VPfit.Chisquared(observed, expected, noise) / freedom


    def plot(self, wavelength_array, flux_array, clouds=None, n=1, onesigmaerror = 0.02, 
            start_pix=None, end_pix=None, filename=None):
        """
        Plot the fitted absorption profile

        Args:
            wavelength_array (numpy array):
            flux_array (numpy array): original flux array, same length as wavelength_array
            filename (string): for saving the plot
	    clouds (pandas dataframe): dataframe containing details on each absorption feature
            n (int): number of *fitted* absorption profiles
            onesigmaerror (float): noise on profile plot
        """

        if not start_pix:
            start_pix = 0
        if not end_pix:
            end_pix = len(wavelength_array)

        f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=False, figsize=(10,10))

        ax1.plot(wavelength_array, (flux_array - self.total.value) / onesigmaerror)
        ax1.hlines(1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax1.hlines(-1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax1.hlines(3, wavelength_array[0], wavelength_array[-1], color='red', linestyles='--')
        ax1.hlines(-3, wavelength_array[0], wavelength_array[-1], color='red', linestyles='--')

        ax2.plot(wavelength_array, flux_array, color='black', linewidth=1.0)

        if clouds is not None:
            for c in range(len(clouds)):
                if c==0:
                    ax2.plot(wavelength_array, Tau2flux(clouds.ix[c]['tau'][start_pix:end_pix]),
                             color="red", label="Actual", lw=1.5)
                else:
                    ax2.plot(wavelength_array, Tau2flux(clouds.ix[c]['tau'][start_pix:end_pix]),
                             color="red", lw=1.5)


        for c in range(n):
            if c==0:
                ax2.plot(wavelength_array, Tau2flux(self.estimated_profiles[c].value),
                         color="green", label="Fit")
            else:
                ax2.plot(wavelength_array, Tau2flux(self.estimated_profiles[c].value),
                         color="green")


        ax2.legend()

        ax3.plot(wavelength_array, flux_array, label="Measured")
        ax3.plot(wavelength_array, self.total.value, color='green', label="Fit", linewidth=2.0)
        ax3.legend()

        f.subplots_adjust(hspace=0)

        if hasattr(self, 'fit_time'): ax1.set_title("Fit time: " + self.fit_time)
        ax1.set_ylabel("Residuals")
        ax2.set_ylabel("Normalised Flux")
        ax3.set_ylabel("Normalised Flux")
        ax3.set_xlabel("$ \lambda (\AA)$")

        if filename:
            plt.savefig(filename)
        else:
            plt.show()


    def find_local_minima(self, f_array, window=101):
        """
        Find the local minima of an absorption profile.

        Args:
            f_array: flux array
            window: smoothing window, pixels
        Returns:
            indices of local minima in flux_array
        """

        # smooth flux profile
        smoothed_flux = savgol_filter(f_array, window, 1)

        return find_peaks_cwt(smoothed_flux * -1, np.array([window/3]))


    def initialise_components(self, frequency_array, n, sigma_max):
        """
        Initialise each fitted component of the model in optical depth space. Each component 
        consists of three variables, height, centroid and sigma. These variables are encapsulated 
        in a deterministic profile variable. The variables are stored in a dictionary, `estimated_variables`, 
        and the profiles in a list, `estimated_profiles`.

        Args:
            frequency_array (numpy array)
            n (int): number of components
            sigma_max (float): maximum permitted range of fitted sigma values
        """

        self.estimated_variables = {}
        self.estimated_profiles = []

        for component in range(n):

            self.estimated_variables[component] = {}

            @mc.stochastic(name='xexp_%d' % component)
            def xexp(value=0.5):
                if value < 0:
                    return -np.inf
                else:
                    return np.log(value * np.exp(-value))

            self.estimated_variables[component]['amplitude'] = xexp
            #self.estimated_variables[component]['height'] = mc.Uniform("est_height_%d" % component, 0, 5)

            self.estimated_variables[component]['centroid'] = mc.Uniform("est_centroid_%d" % component,
                                                                         frequency_array[0], frequency_array[-1])

            self.estimated_variables[component]['sigma'] = mc.Uniform("est_sigma_%d" % component, 0, sigma_max)

            @mc.deterministic(name='component_%d' % component, trace = True)
            def profile(x=frequency_array,
                        centroid=self.estimated_variables[component]['centroid'],
                        sigma=self.estimated_variables[component]['sigma'],
                        height=self.estimated_variables[component]['amplitude']):

                return self.GaussFunction(x, height, centroid, sigma )

            self.estimated_profiles.append(profile)


    def initialise_voigt_profiles(self, frequency_array, n, sigma_max, local_minima=[]):
        """
        Args:
            frequency_array (numpy array)
            n (int): number of components
            sigma_max (float): maximum permitted range of fitted sigma values
        """

        #if n < len(local_minima):
        #    raise ValueError("Less profiles than number of minima.")

        self.estimated_variables = {}
        self.estimated_profiles = []

        for component in range(n):

            self.estimated_variables[component] = {}

            @mc.stochastic(name='xexp_%d' % component)
            def xexp(value=0.5):
                if value < 0:
                    return -np.inf
                else:
                    return np.log(value * np.exp(-value))

            self.estimated_variables[component]['amplitude'] = xexp

            self.estimated_variables[component]['centroid'] = mc.Uniform("est_centroid_%d" % component,
                                                                            frequency_array[0], frequency_array[-1])


            self.estimated_variables[component]['L_fwhm'] = mc.Uniform("est_L_%d" % component, 0, sigma_max)
            self.estimated_variables[component]['G_fwhm'] = mc.Uniform("est_G_%d" % component, 0, sigma_max)

            @mc.deterministic(name='component_%d' % component, trace = True)
            def profile(x=frequency_array,
                        centroid=self.estimated_variables[component]['centroid'],
                        amplitude=self.estimated_variables[component]['amplitude'],
                        L=self.estimated_variables[component]['L_fwhm'],
                        G=self.estimated_variables[component]['G_fwhm']):
                return self.VoigtFunction(x, centroid, amplitude, L, G)

            self.estimated_profiles.append(profile)


    def initialise_model(self, frequency, flux, n, local_minima=[], voigt=False):
        """
        Initialise deterministic model of all absorption features, in normalised flux.

        Args:
            frequency (numpy array)
            flux (numpy array): flux values at each wavelength
            n (int): number of absorption profiles to fit
        """
        
        self.sigma_max = (frequency[-1] - frequency[0]) / 2.

        # always reinitialise profiles, otherwise starts sampling from previously calculated parameter values.
        if(voigt):
            if self.verbose:
                print "Initialising Voigt profile components."
            self.fwhm_max = self.sigma_max * 2*np.sqrt(2*np.log(2.))
            self.initialise_voigt_profiles(frequency, n, self.fwhm_max, local_minima=local_minima)
        else:
            if self.verbose:
                print "Initialising Gaussian profile components."
            self.initialise_components(frequency, n, self.sigma_max)

        # deterministic variable for the full profile, given in terms of normalised flux
        @mc.deterministic(name='profile', trace=False)
        def total(profile_sum=self.estimated_profiles):
            return Tau2flux(sum(profile_sum))

        self.total = total

        # represent full profile as a normal distribution
        self.profile = mc.Normal("obs", self.total, self.std_deviation, value=flux, observed=True)

        # create model with parameters of all profiles to be fitted
        # TODO: fix the deprecation error (this is the only case where a "Model" is instantiated):
        # /home/jacobc/anaconda2/envs/vamp27pymc237/lib/python2.7/site-packages/pymc/MCMC.py:81: 
        # UserWarning: Instantiating a Model object directly is deprecated. We recommend passing variables directly to the Model subclass.  warnings.warn(message)

        self.model = mc.Model([self.estimated_variables[x][y] for x in self.estimated_variables \
                            for y in self.estimated_variables[x]])# + [std_deviation])


    def map_estimate(self, iterations=2000):
        """
        Compute the Maximum A Posteriori estimates for the initialised model
        """

        self.map = mc.MAP(self.model)
        self.map.fit(iterlim=iterations, tol=1e-3)

        
    def mcmc_fit(self, iterations=15000, burnin=100, thinning=15, step_method=mc.AdaptiveMetropolis):
        """
        MCMC fit of `n` absorption profiles to a given spectrum

        Args:
            iterations (int): How long to sample the MCMC for
            burnin (int): How many iterations to ignore from the start of the chain
            thinning (int): How many iterations to ignore each tally
            step_method (PyMC steo method object)
        """

        try:
            getattr(self, 'map')
        except AttributeError:
            print "\nWARNING: MAP estimate not provided. \nIt is recommended to compute this \
            in advance of running the MCMC so as to start the sampling with good initial values."

        # create MCMC object
        self.mcmc = mc.MCMC(self.model)

        # change step method
        if step_method != mc.AdaptiveMetropolis:
            print "Changing step method for each parameter."
            for index, item in self.estimated_variables.items():
                self.mcmc.use_step_method(step_method, item['sigma'])
                self.mcmc.use_step_method(step_method, item['centroid'])
                self.mcmc.use_step_method(step_method, item['height'])
        else:
            print "Using Adaptive Metropolis step method for each parameter."

        # fit the model
        starttime=datetime.datetime.now()
        self.mcmc.sample(iter=iterations, burn=burnin, thin=thinning)
        self.fit_time = str(datetime.datetime.now() - starttime)
        print "\nTook:", self.fit_time, " to finish."


    def find_bic(self, frequency_array, flux_array, n, noise_array, freedom, voigt=False, 
                iterations=3000, thin=15, burn=300, thorough=False):
        """
        Initialise the Voigt model and run the MCMC fitting for a particular number of 
        regions and return the Bayesian Information Criterion. Used to identify the 
        most appropriate number of profles in the region.
        
        Args:
            frequency_array (numpy array)
            flux_array (numpy array)
            n (int): number of minima to fit in the region
            noise_array (numpy array)
            freedom (int): number of degrees of freedom
            voigt (Boolean): switch to fit as Voigt profile or Gaussian
            iterations, thin, burn (ints): MCMC parameters
        """
        
        self.bic_array = []
        self.red_chi_array = []
        for i in range(3):
            self.initialise_model(frequency_array, flux_array, n, voigt=voigt)
            self.map = mc.MAP(self.model)
            self.mcmc = mc.MCMC(self.model)
            if thorough:
                self.map.fit(iterlim=iterations, tol=1e-3)
                self.mcmc.sample(iter=iterations, burn=burn, thin=thin, progress_bar=False)
                self.map.fit(iterlim=iterations, tol=1e-3)
            self.mcmc.sample(iter=iterations, burn=burn, thin=thin, progress_bar=False)
            self.map.fit(iterlim=iterations, tol=1e-3)
            self.bic_array.append(self.map.BIC)
            self.red_chi_array.append(self.ReducedChisquared(flux_array, self.total.value, noise_array, freedom))
        return


    def chain_covariance(self, n, voigt=False):
        """
        Find the covariance matrix of the MCMC chain variables
        Args:
            n (int): the number of lines in the region
            voigt (boolean): switch to use Voigt parameters
        Returns:
            cov (numpy array): the covariance matrix
            cov =   [ var(amplitude),           cov(amplitude, sigma),  cov(amplitude, centroid) ]
                    [ cov(sigma, amplitude),    var(sigma),             cov(sigma, centroid)     ]
                    [ cov(centroid, amplitude), cov(centroid, sigma),   var(centroid)            ]

        """
        cov = np.zeros((n, 3, 3))
        for i in range(n):
            amp_samples = self.mcmc.trace('xexp_'+str(i))[:]
            if not voigt:
                sigma_samples = self.mcmc.trace('est_sigma_'+str(i))[:]
            elif voigt:
                gfwhm_samples = self.mcmc.trace('est_G_'+str(i))[:]
                sigma_samples = self.GaussianWidth(gfwhm_samples)
            c_samples = self.mcmc.trace('est_centroid_'+str(i))[:]
        
            cov[i] = np.cov(np.array((amp_samples, sigma_samples, c_samples)))
        return cov


def compute_detection_regions(wavelengths, fluxes, noise, min_region_width=2, 
                            N_sigma=4.0, extend=False):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength    
        noise (numpy array): noise value at each wavelength 
        min_region_width (int): minimum width of a detection region (pixels)
        N_sigma (float): detection threshold (std deviations)
        extend (boolean): default is False. Option to extend detected regions untill tau
                        returns to continuum.

    Returns:
        regions_l (numpy array): contains subarrays with start and end wavelengths
        regions_i (numpy array): contains subarrays with start and end indices
    """
    print('Computing detection regions...')

    num_pixels = len(wavelengths)
    pixels = range(num_pixels)
    min_pix = 1
    max_pix = num_pixels - 1

    flux_ews = [0.] * num_pixels
    noise_ews = [0.] * num_pixels
    det_ratio = [-float('inf')] * num_pixels

    # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
    for i in range(min_pix, max_pix):
        flux_dec = 1.0 - fluxes[i]
        if flux_dec < noise[i]:
            flux_dec = 0.0
        flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
        noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

    # dev: no need to set end values = 0. since loop does not set end values
    flux_ews[0] = 0.
    noise_ews[0] = 0.

    # Range of standard deviations for Gaussian convolution
    std_min = 2
    std_max = 11

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    xarr = np.array([p - (num_pixels-1)/2.0 for p in range(num_pixels)])
    
    # this part can remain the same, since it uses EW in wavelength units, not flux
    for std in range(std_min, std_max):

        gaussian = VPfit.GaussFunction(xarr, 1.0, 0.0, std)

        flux_func = np.convolve(flux_ews, gaussian, 'same')
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), 'same')

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    for i in range(num_pixels):
        if start == 0 and det_ratio[i] > N_sigma and fluxes[i] < 1.0:
            start = i
        elif start != 0 and (det_ratio[i] < N_sigma or fluxes[i] > 1.0):
            if (i - start) > min_region_width:
                end = i
                region_endpoints.append([start, end])
            start = 0

    # made extend a kwarg option
    # lines may not go down to 0 again before next line starts
    
    if extend:
        # Expand edges of region until flux goes above 1
        regions_expanded = []
        for reg in region_endpoints:
            start = reg[0]
            i = start
            while i > 0 and fluxes[i] < 1.0:
                i -= 1
            start_new = i
            end = reg[1]
            j = end
            while j < (len(fluxes)-1) and fluxes[j] < 1.0:
                j += 1
            end_new = j
            regions_expanded.append([start_new, end_new])

    else: regions_expanded = region_endpoints

    # Change to return the region indices
    # Combine overlapping regions, check for detection based on noise value
    # and extend each region again by a buffer
    regions_l = []
    regions_i = []
    buffer = 3
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        #TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
        if i<(len(regions_expanded)-1) and end > regions_expanded[i+1][0]:
            end = regions_expanded[i+1][1]
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if flux_dec > abs(noise[j]) * N_sigma:
                if start >= buffer:
                    start -= buffer
                if end < len(wavelengths) - buffer:
                    end += buffer
                regions_l.append([wavelengths[start], wavelengths[end]])
                regions_i.append([start, end])
                break

    print('Found {} detection regions.'.format(len(regions_l)))
    return np.array(regions_l), np.array(regions_i)


def estimate_n(flux_array):
    """
    Make initial guess for number of local minima in the region.
        
    Smooth the spectra with a gaussian and find the number of local minima.
    as a safety precaucion, set the initial guess for number of profiles to 1 if
    there are less than 4 local minima.

    Args:
        flux_array (numpy array)
    """
        
    n = argrelextrema(gaussian_filter(flux_array, 3), np.less)[0].shape[0]
    if n < 4:
        n = 1
    return n


def region_fit(frequency_array, flux_array, n, noise_array, freedom, voigt=False, 
                chi_limit=1., verbose=True, iterations=3000, thin=15, burn=300):
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
        print "Setting initial number of lines to: {}".format(n)
    while not finished:
        vpfit = VPfit()
        vpfit.find_bic(frequency_array, flux_array, n, noise_array, freedom, voigt=voigt, iterations=iterations, burn=burn)
        if first:
            first = False
            n += 1
            bic_old = vpfit.bic_array[-1]
            vpfit_old = copy(vpfit)
            del vpfit
        else:
            if bic_old > np.average(vpfit.bic_array):
                if verbose:
                    print "Old BIC value of {:.2f} is greater than the current {:.2f}.".format(bic_old, np.average(vpfit.bic_array))
                bic_old = np.average(vpfit.bic_array)
                vpfit_old = copy(vpfit)
                if np.average(vpfit.red_chi_array) < chi_limit:
                    if verbose:
                        print "Reduced Chi squared is less than {}".format(chi_limit)
                        print "Final n={}".format(n)
                    finished = True
                    continue
                n += 1
                del vpfit
                if verbose:
                    print "Increasing the number of lines to: {}".format(n)
            else:
                if verbose:
                    print "BIC increased with increasing the line number, stopping."
                    n -= 1
                    print "Final n={}.".format(n)
                finished = True
                continue
    gc.collect()
    return vpfit_old



def fit_spectrum(wavelength_array, noise_array, flux_array, line, voigt=False, chi_limit=1.5, folder=None,
                 mcmc_cov=False, get_mcmc_err=True, convergence_attempts=10):
    """
    The main function. Takes an input spectrum, splits it into manageable regions, and fits 
    the individual regions using PyMC. Finally calculates the Doppler parameter b, the 
    column density N and the equivalent width and the centroids of the absorption lines.

    Args:
        wavelength_array (numpy array): in Angstroms
        noise_array (numpy array)
        flux_array(numpy array)
        flux_array (numpy array)
        line (float): the rest wavelength of the absorption line in Angstroms
        voigt (boolean): switch to fit Voigt profile instead of Gaussian. Default: True.
        folder (string): if plotting the fits and saving them, provide a directory. Default: None.
        mcmc_cov (boolean): switch to do error propogation from the mcmc chain. Default: False
        #get_b_std (boolean): switch to find the standard deviation of b from the mcmc chain. Default: True.
        get_mcmc_err (boolean): switch to find the standard deviation (of b and N) from the mcmc chain. Default: True

    Returns:
        b (numpy array): Doppler parameter, in km/s.
        N (numpy array): Column density, in m**-2
        ew (numpy array): Equivalent widths, in Angstroms.
        centers (numpy array): Centroids of absorption lines, in Angstroms.
    """
    chi_sq_maximum = 10 #the maximum "acceptable" chi-squared value before the line adder forcer kicks in

    frequency_array = Wave2freq(wavelength_array)
    #TODO: see if I actually need this? doesn't appear so, since I don't see anywhere that flux --> optical depth?
    #flux_floor = 1e-6
    #flux_array = flux_array.clip(min=flux_floor) #set all negative fluxes to some very small value

    # identify regions to fit in the spectrum
    regions, region_pixels = compute_detection_regions(wavelength_array, 
                            flux_array, noise_array, min_region_width=2)

    params = {'b': np.array([]), 'b_std': np.array([]), 'N': np.array([]), 'N_std': np.array([]), 
                'EW': np.array([]), 'centers': np.array([]), 'region_numbers': np.array([])}

    flux_model = {'total': np.ones(len(flux_array)), 'chi_squared': np.zeros(len(regions)), 'region_pixels': region_pixels,
                'amplitude': np.array([]), 'sigmas': np.array([]), 'centers': np.array([]), 'region_numbers': np.array([]),
                'std_a': np.array([]), 'std_s': np.array([]), 'std_c': np.array([]), 'cov_as': np.array([])}
    

    max_single_region_components = 15
    ideal_single_region_components = 5
    if (len(regions) == 1):
        start = region_pixels[0][0]
        end = region_pixels[0][1]
        fluxes = np.flip(flux_array[start:end], 0)
        n = estimate_n(fluxes)
        if (n > max_single_region_components):
            print(str(n) + " components should be in more than 1 region!")
            #TODO: split the regions up
            #TODO: find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled
            forced_number_regions = n // ideal_single_region_components

            #TODO: turn this index gathering into a function? just in case it needs to be repeated
            ind = np.argpartition(fluxes, 10*forced_number_regions)[-10*forced_number_regions:]
            ind_sorted = np.flip(ind[np.argsort(fluxes[ind])], axis=0) #indicies sorted from highest flux to lowest flux

            #don't let regions be less than X% (5%?) of the spectrum
            num_pixels = len(fluxes)
            min_region_percentage = 5
            min_region_size = num_pixels / min_region_percentage


            #new_starts = [start]
            #new_ends = []
            splitting_points = [start, end]
            # original "start" is the beginning of the 1st region
            # original "end" is the end of the last region
            # each region will be contiguous, so need (forced_number_regions - 1) indexes
            # these indexes will have to be at least <min_region_size> away from each other.

            for i in range(len(ind_sorted)):
                # go through indices of maximum flux, in descending order
                # see if they can be the required distance away from each other.
                # if they can't be made to work, try working down the list of maximum fluxes
                if (len(splitting_points) == (forced_number_regions+1)):
                    break #stop once enough points to split have been found
                else:
                    dist_is_fine = True
                    for j in range(len(splitting_points)):
                        # check the distance between a possible "splitting point" and the existing splitting points
                        dist = abs(ind_sorted[i] - splitting_points[j])
                        if (dist < min_region_size):
                            dist_is_fine = False

                    if dist_is_fine: #if the region would be large enough, then split along this point
                        splitting_points.append(ind_sorted[i])




    j = 0
    for start, end in region_pixels:
        fluxes = np.flip(flux_array[start:end], 0)
        noise = np.flip(noise_array[start:end], 0)
        waves = np.flip(wavelength_array[start:end], 0)
        nu = np.flip(frequency_array[start:end], 0)


        best_chi_squared = -1 #initialize best_chi_sq

        #track the chi-squared values and associated number of components, to force-add components if necessary
        attempt_n = []
        attempt_chi_squareds = []
        for _ in range(convergence_attempts):

            # make initial guess for number of lines in a region
            n = estimate_n(fluxes)

            # force the fitter to add additional components if it has previously failed with fewer components
            # (even if BIC indicates otherwise)

            n_is_sensible = False
            while(n_is_sensible == False):
                if (attempt_n.count(n) > 2): #check if it's failed to converge for a few attempts with this number of components
                    #check what the chi-squareds were for this number
                    ii = [i for i,val in enumerate(attempt_n) if val==n]
                    #print(ii)
                    #print(attempt_n)
                    #print(attempt_chi_squareds)
                    chi_squareds = np.array(attempt_chi_squareds)[ii]
                    if (np.min(chi_squareds) > chi_sq_maximum): #if the best chi squared value is still too high
                        print("Chi-squareds have been too high for fits using n=" + str(n) + " components. (chi-sq>" + str(chi_sq_maximum) + ")")
                        n += 1 #force it to add another component
                        print("Forcing the fitter to increase the fit to at least n=" + str(n) + " components.")
                    else:
                        n_is_sensible = True
                else:
                    n_is_sensible = True



            # number of degrees of freedom = number of data points + number of parameters
            freedom = len(fluxes) - 3*n
            
            # fit the region by minimising BIC and chi squared
            fit = region_fit(nu, fluxes, n, noise, freedom, voigt=voigt, chi_limit=chi_limit)
            
            # evaluate overall chi squared
            n = len(fit.estimated_profiles)
            freedom = len(fluxes) - 3*n

            current_chi_squared = fit.ReducedChisquared(fluxes, fit.total.value, noise, freedom)
            attempt_n.append(n)
            attempt_chi_squareds.append(current_chi_squared)

            print 'Reduced chi squared is {:.2f}'.format(current_chi_squared)

            #if this is the best (or first) chi-squared value seen so far, record it and save the fit
            if (current_chi_squared < best_chi_squared) or (best_chi_squared == -1) :
                best_chi_squared = current_chi_squared
                best_fit = fit

            # if chi squared is sufficiently small, stop there. If not, repeat the region fitting
            if best_chi_squared < chi_limit:
                break

        # use the best fit found in the loop above
        fit = best_fit
        n = len(fit.estimated_profiles)
        flux_model['chi_squared'][j] = best_chi_squared
        print("Using model with best chi-squared seen, which is : {:.2f}".format(best_chi_squared))


        print '\n'
        flux_model['total'][start:end] = np.flip(fit.total.value, 0)

        flux_model['region_'+str(j)+'_wave'] = np.flip(waves, 0)


        flux_model['region_'+str(j)+'_flux'] = np.ones((n, len(fluxes)))
        for k in range(n):
            flux_model['region_'+str(j)+'_flux'][k] = np.flip(Tau2flux(fit.estimated_profiles[k].value), 0)

        heights = np.array([fit.estimated_variables[i]['amplitude'].value for i in range(n)])
        centers = np.array([Freq2wave(fit.estimated_variables[i]['centroid'].value) for i in range(n)])
        #region_numbers = np.array(j for i in range(n)
        region_numbers = np.array([j for i in range(n)])


        if not voigt:
            sigmas = np.array([fit.estimated_variables[i]['sigma'].value for i in range(n)])
        
        elif voigt:
            g_fwhms = np.array([fit.estimated_variables[i]['G_fwhm'].value for i in range(n)])
            sigmas = VPfit.GaussianWidth(g_fwhms)
        
        flux_model['amplitude'] = np.append(flux_model['amplitude'], heights)
        flux_model['centers'] = np.append(flux_model['centers'], centers)
        flux_model['region_numbers'] = np.append(flux_model['region_numbers'], region_numbers)
        flux_model['sigmas'] = np.append(flux_model['sigmas'], sigmas)

        if mcmc_cov:

            cov = fit.chain_covariance(n, voigt=voigt)
            std_a = np.sqrt([cov[i][0][0] for i in range(n)])
            std_s = np.sqrt([cov[i][1][1] for i in range(n)])
            std_c = np.sqrt([cov[i][2][2] for i in range(n)])
            cov_as = np.array([cov[i][0][1] for i in range(n)])

            flux_model['std_a'] = np.append(flux_model['std_a'], std_a)
            flux_model['std_s'] = np.append(flux_model['std_s'], std_s)
            flux_model['std_c'] = np.append(flux_model['std_c'], std_c)
            flux_model['cov_as'] = np.append(flux_model['cov_as'], cov_as)

            params['N_std'] = np.append(params['N_std'], ErrorN(heights, sigmas, std_a, std_s, cov_as))
        
        elif get_mcmc_err:
            stats = fit.mcmc.stats()
            #TODO: figure out why this "std_s = " line is giving " KeyError: 'est_sigma_0' " when --voigt is used
            # (probably because it's called something else in the Voigt object?)
            """
             Traceback (most recent call last):
                 File "/home/jacobc/VAMP/vpfits.py", line 1082, in <module>
                     params, flux_model = fit_spectrum(wavelength, noise, flux, args.line, voigt=args.voigt, folder=args.output_folder)
                 File "/home/jacobc/VAMP/vpfits.py", line 905, in fit_spectrum
                     std_s = np.array([stats['est_sigma_'+str(i)]['standard deviation'] for i in range(n)])
             KeyError: 'est_sigma_0'
             """
            std_s = np.array([stats['est_sigma_'+str(i)]['standard deviation'] for i in range(n)]) #mcmc fitting error for "sigmas"
            std_a = np.array([stats['xexp_'+str(i)]['standard deviation'] for i in range(n)])      #mcmc fitting error for "amplitude" (referred to as "xexp"

            flux_model['std_s'] = np.append(flux_model['std_s'], std_s)
            flux_model['std_a'] = np.append(flux_model['std_a'], std_a)

            params['b_std'] = np.append(params['b_std'], ErrorB(std_s, line))
            covariance = [0] * len(std_a) # treat covariance as 0  TODO: figure out how to extract covariance (if possible) from MCMC chain.
            params['N_std'] = np.append(params['N_std'], ErrorN(amplitude=heights,sigma=sigmas,std_a=std_a, std_s=std_s, cov_as=covariance))

        params['b'] = np.append(params['b'], DopplerParameter(sigmas, line))
        params['N'] = np.append(params['N'], ColumnDensity(heights, sigmas))
        params['centers'] = np.append(params['centers'], centers)
        params['region_numbers'] = np.append(params['region_numbers'], region_numbers)
        for k in range(n):
            params['EW'] = np.append(params['EW'], EquivalentWidth(fit.estimated_profiles[k].value, [waves[0], waves[-1]]))
        
        j += 1

    if folder:
        plot_spectrum(wavelength_array, flux_array, flux_model, region_pixels, folder)

    """
    print("len(N) in params: " + str(len(params['N'])))
    print("len(centers) in params: " + str(len(params['centers'])))
    print("len(region_numbers) in params: " + str(len(params['region_numbers'])))
    print("len(centers) in flux_model: " + str(len(flux_model['centers'])))
    print("len(region_numbers) in flux_model: " + str(len(flux_model['region_numbers'])))
    """
    return params, flux_model


def plot_spectrum(wavelength_array, flux_data, flux_model, regions, folder):
    """
    Routine to plot the fits to the data. First plot is the total fit, second is the component
    Voigt profiles, third is the residuals.
    Args:
        wavelength_array (numpy array)
        flux_data (numpy array): data values for flux 
        flux_model (dict): dict output of the fitting routine, containing profiles for each 
                            component
        regions  (numpy array): pixel boundaries for the regions of the spectrum
        folder (string): a directory for saving the plots
    """

    def plot_bracket(x, axis, dir):
        height = .2
        arm_length = 0.1
        axis.plot((x, x), (1-height/2, 1+height/2), color='magenta')

        if dir=='left':
            xarm = x+arm_length
        if dir=='right':
            xarm = x-arm_length

        axis.plot((x, xarm), (1-height/2, 1-height/2), color='magenta')
        axis.plot((x, xarm), (1+height/2, 1+height/2), color='magenta')

    model = flux_model['total']

    N = 6
    length = len(flux_data) / N

    fig, ax = plt.subplots(N, figsize=(15,15))
    for n in range(N):
        lower_lim = n*length
        upper_lim = n*length+length

        ax[n].plot(wavelength_array, flux_data, c='black', label='Measured')
        ax[n].plot(wavelength_array, model, c='green', label='Fit')
        ax[n].set_xlim(wavelength_array[lower_lim], wavelength_array[upper_lim])
        for (start, end) in regions:
            plot_bracket(wavelength_array[start], ax[n], 'left')
            plot_bracket(wavelength_array[end], ax[n], 'right')

    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.savefig(folder+'fit.png')
    plt.clf()

    fig, ax = plt.subplots(N, figsize=(15,15))
    for n in range(N):
        lower_lim = n*length
        upper_lim = n*length+length

        for i in range(len(regions)):
            start, end = regions[i]
            region_data = flux_model['region_'+str(i)+'_flux']
            for j in range(len(region_data)):
                ax[n].plot(wavelength_array[start:end], region_data[j], c='green')
            plot_bracket(wavelength_array[start], ax[n], 'left')
            plot_bracket(wavelength_array[end], ax[n], 'right')

        ax[n].set_xlim(wavelength_array[lower_lim], wavelength_array[upper_lim])        

    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.savefig(folder+'components.png')
    plt.clf()    

    fig, ax = plt.subplots(N, figsize=(15,15))
    for n in range(N):
        lower_lim = n*length
        upper_lim = n*length+length

        ax[n].plot(wavelength_array, (flux_data - model), c='blue')
        ax[n].hlines(1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax[n].hlines(-1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax[n].hlines(3, wavelength_array[0], wavelength_array[-1], color='red', linestyles='--')
        ax[n].hlines(-3, wavelength_array[0], wavelength_array[-1], color='red', linestyles='--')
        ax[n].set_xlim(wavelength_array[lower_lim], wavelength_array[upper_lim])
        for (start, end) in regions:
            plot_bracket(wavelength_array[start], ax[n], 'left')
            plot_bracket(wavelength_array[end], ax[n], 'right')

    plt.xlabel('Wavelength (A)')
    plt.ylabel('Flux')
    plt.savefig(folder+'residuals.png')
    plt.clf()

    return

def write_file(params, filename, format):
    """
    Save file with physical parameters from fit
    Args:
        params (dict): output of fit_spectrum
        filename (string): name of ascii file
        format (string): file format to save. Can be ascii or h5.
    """
    if format == 'ascii':
        import astropy.io.ascii as ascii
        if 'N' in params:
            formats = {'N': '%.6g', 'N_std': '%.6g', 'EW': '%.6g', 'b': '%.6g', 'b_std': '%.6g', 'centers': '%.6g',
                       'region_numbers': '%.6g'}
            ascii.write(params, filename, formats=formats)
        elif 'total' in params:
            formats = {'total': '%.6g', 'chi_squared': '%.6g', 'centers': '%.6g', 'amplitude': '%.6g', 'sigmas': '%.6g', 
                        'std_a': '%.6g', 'std_s': '%.6g', 'std_c': '%.6g', 'cov_as': '%.6g'}
            ascii.write(params, filename, formats=formats)
    if format == 'h5':
        with h5py.File(filename, 'a') as f:
            for p in params.keys():
                f.create_dataset(p, data=np.array(params[p]))


if __name__ == "__main__":  

    import argparse
    parser = argparse.ArgumentParser(description='Voigt Automatic MCMC Profiles (VAMP)')
    parser.add_argument('data_file',
                        help='Input file with absorption spectrum data from Pygad.')
    parser.add_argument('line',
                        help='Wavelength of the absorption line in Angstroms.', 
                        type=float)
    parser.add_argument('--output_folder',
                        help='Folder to save output.', default='./')
    parser.add_argument('--voigt',
                        help='Fit Voigt profile. Default: False', 
                        action='store_true')
    args = parser.parse_args()

    name = args.data_file.split('/', -1)[-1]
    name = name[:name.find('.')]

    if args.voigt == True:
        args.output_folder += name + '_voigt_'
    else:
        args.output_folder += name + '_gauss_'

    #onesigmaerror = 0.02
    #noise = np.random.normal(0.0, onesigmaerror, len(wavelength_array))
    import h5py
    data = h5py.File(args.data_file, 'r')
    
    wavelength = np.array(data['wavelength'][:])
    noise = np.array(data['noise'][:])
    flux = np.array(data['flux'][:])

    convergence_attempts = 10 #TODO: set this in some configuration file

    params, flux_model = fit_spectrum(wavelength, noise, flux, args.line, voigt=args.voigt, folder=args.output_folder,
                                      convergence_attempts=convergence_attempts)
    write_file(params, args.output_folder+'params.h5', 'h5')
    write_file(flux_model, args.output_folder+'flux_model.h5', 'h5')

