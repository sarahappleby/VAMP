"""
VPfit

Fit Voigt Profiles using MCMC. Uses bayesian model selection to pick the appropriate number 
of profiles for a given absorption.

The main class containing the fitting functionality is `VPfit`.

"""

import matplotlib.pyplot as plt

import datetime

import numpy as np
import pymc as mc

# Voigt modules
# from scipy.special import wofz
from astropy.modeling.models import Voigt1D

# peak finding modules
from scipy.signal import find_peaks_cwt
from scipy.signal import savgol_filter

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
                print("Initialising Voigt profile components.")
            self.fwhm_max = self.sigma_max * 2*np.sqrt(2*np.log(2.))
            self.initialise_voigt_profiles(frequency, n, self.fwhm_max, local_minima=local_minima)
        else:
            if self.verbose:
                print("Initialising Gaussian profile components.")
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
            print("\nWARNING: MAP estimate not provided. \nIt is recommended to compute this \
            in advance of running the MCMC so as to start the sampling with good initial values.")

        # create MCMC object
        self.mcmc = mc.MCMC(self.model)

        # change step method
        if step_method != mc.AdaptiveMetropolis:
            print("Changing step method for each parameter.")
            for index, item in self.estimated_variables.items():
                self.mcmc.use_step_method(step_method, item['sigma'])
                self.mcmc.use_step_method(step_method, item['centroid'])
                self.mcmc.use_step_method(step_method, item['height'])
        else:
            print("Using Adaptive Metropolis step method for each parameter.")

        # fit the model
        starttime=datetime.datetime.now()
        self.mcmc.sample(iter=iterations, burn=burnin, thin=thinning)
        self.fit_time = str(datetime.datetime.now() - starttime)
        print("\nTook:", self.fit_time, " to finish.")


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