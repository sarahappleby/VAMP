"""
VPfit

Fit Voigt Profiles using MCMC. Uses bayesian model selection to pick the appropriate number of profiles for a given absorption.

The main class containing the fitting functionality is `VPfit`. A mock absorption generator, `mock_absorption`, is also included for demonstration.

See `__init__` for example usage.

Todo:
* Add Voigt profile
"""

import random
import datetime

import numpy as np
import pandas as pd

import pymc as mc

import matplotlib.pylab as pylab


class VPfit():

    def __init__(self):
        """
        Initialise noise variable
        """
        self.std_deviation = 1./mc.Uniform("sd", 0, 1)**2

    @staticmethod
    def GaussFunction(wavelength_array, amplitude, centroid, sigma):
        """
        Gaussian.

        Args:
            wavelength_array (numpy array)
            amplitude (float)
            centroid (float): must be between the limits of wavelength_array
            sigma (float)
        """
        return amplitude * np.exp(-0.5 * ((wavelength_array - centroid) / sigma) ** 2)


    def Absorption(self, arr):
        """
        Convert optical depth in to absorption profile

        Args:
            arr (numpy array): array of optical depth values
        """

        return np.exp(-arr)


    @staticmethod
    def Chisquared(observed, expected):
        return sum(((observed - expected)**2) / expected)


    @staticmethod
    def ReducedChisquared(observed, expected, freedom):
        return VPfit.Chisquared(observed, expected) / (len(expected) - freedom)


    def plot(self, wavelength_array, flux_array, clouds=None, n=1, onesigmaerror = 0.02, start_pix=None, end_pix=None):
        """
        Plot the fitted absorption profile

        Args:
            wavelength_array (numpy array):
            flux_array (numpy array): original flux array, same length as wavelength_array
            clouds (pandas dataframe): dataframe containing details on each absorption feature
            n (int): number of *fitted* absorption profiles
            onesigmaerror (float): noise on profile plot
        """

        if not start_pix:
            start_pix = 0
        if not end_pix:
            end_pix = len(wavelength_array)

        f, (ax1, ax2, ax3) = pylab.subplots(3, sharex=True, sharey=False, figsize=(10,10))

        ax1.plot(wavelength_array, (flux_array - self.total.value) / onesigmaerror)
        ax1.hlines(1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax1.hlines(-1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')

        ax2.plot(wavelength_array, flux_array, color='black', linewidth=1.0)

        if clouds is not None:
            for c in range(len(clouds)):
                if c==0:
                    ax2.plot(wavelength_array, self.Absorption(clouds.ix[c]['tau'][start_pix:end_pix]),
                             color="red", label="Actual", lw=1.5)
                else:
                    ax2.plot(wavelength_array, self.Absorption(clouds.ix[c]['tau'][start_pix:end_pix]),
                             color="red", lw=1.5)


        for c in range(n):
            if c==0:
                ax2.plot(wavelength_array, self.Absorption(self.estimated_profiles[c].value),
                         color="green", label="Fit")
            else:
                ax2.plot(wavelength_array, self.Absorption(self.estimated_profiles[c].value),
                         color="green")


        ax2.legend()

        ax3.plot(wavelength_array, flux_array, label="Measured")
        ax3.plot(wavelength_array, self.total.value, color='green', label="Fit", linewidth=2.0)
        ax3.legend()

        f.subplots_adjust(hspace=0)

        ax1.set_title("Fit time: " + self.fit_time)
        ax1.set_ylabel("Residuals")
        ax2.set_ylabel("Normalised Flux")
        ax3.set_ylabel("Normalised Flux")
        ax3.set_xlabel("$ \lambda (\AA)$")

        pylab.show()

    def initialise_components(self, wavelength_array, n, sigma_max = 5):
        """
        Initialise each fitted component of the model in optical depth space. Each component consists of three variables, height, centroid and sigma. These variables are encapsulated in a deterministic profile variable. The variables are stored in a dictionary, `estimated_variables`, and the profiles in a list, `estimated_profiles`.

        Args:
            wavelength_array (numpy array)
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

            self.estimated_variables[component]['height'] = xexp
            #self.estimated_variables[component]['height'] = mc.Uniform("est_height_" + str(component), 0, 5)

            self.estimated_variables[component]['centroid'] = mc.Uniform("est_centroid_" + str(component),
                                                                         wavelength_array[0], wavelength_array[-1])

            self.estimated_variables[component]['sigma'] = mc.Uniform("est_sigma_" + str(component), 0, sigma_max)

            @mc.deterministic(name='component_%d' % component, trace = True)
            def profile(x=wavelength_array,
                        centroid=self.estimated_variables[component]['centroid'],
                        sigma=self.estimated_variables[component]['sigma'],
                        height=self.estimated_variables[component]['height']):
                return self.GaussFunction( x, height, centroid, sigma )

            self.estimated_profiles.append(profile)


    def initialise_model(self, wavelength, flux, n):
        """
        Initialise deterministic model of all absorption features, in normalised flux.

        Args:
            wavelength (numpy array)
            flux (numpy array): flux values at each wavelength
            n (int): number of absorption profiles to fit
        """

        # always reinitialise profiles, otherwise starts sampling from previously calculated parameter values.
        self.initialise_components(wavelength, n)

        # deterministic variable for the full profile, given in terms of normalised flux
        @mc.deterministic(name='profile', trace=False)
        def total(profile_sum=self.estimated_profiles):
            return self.Absorption(sum(profile_sum))

        self.total = total

        # represent full profile as a normal
        self.profile = mc.Normal("obs", self.total, self.std_deviation, value=flux, observed=True)

        # create model with parameters of all profiles to be fitted
        self.model = mc.Model([self.estimated_variables[x][y] for x in self.estimated_variables for y in self.estimated_variables[x]])# + [std_deviation])



    def map_estimate(self):
        """
        Compute the Maximum A Posteriori estimates for the initialised model
        """

        self.MAP = mc.MAP(self.model)
        self.MAP.fit()


    def mcmc_fit(self, iterations=10000, burnin=6000, thinning=2):
        """
        MCMC fit of `n` absorption profiles to a given spectrum

        Args:
            wavelength (numpy array)
            flux (numpy array): flux values at each wavelength
            n (int): number of absorption profiles to fit
        """

        try:
            getattr(self, 'MAP')
        except AttributeError:
            print "\nWARNING: MAP estimate not provided. \nIt is recommended to compute this in advance of running the MCMC so as to start the sampling with good initial values."

        # create MCMC object
        self.mcmc = mc.MCMC(self.model)

        # fit the model
        starttime=datetime.datetime.now()
        self.mcmc.sample(iter=iterations, burn=burnin, thin=thinning)
        self.fit_time = str(datetime.datetime.now() - starttime)
        print "\nTook:", self.fit_time, " to finish."



def compute_detection_regions(wavelengths, fluxes, noise, min_region_width=5):
    """
    Finds detection regions above some detection threshold and minimum width.

    Args:
        wavelengths (numpy array)
        fluxes (numpy array): flux values at each wavelength
        noise (numpy array): noise value at each wavelength
        buffer (int): buffer before combining close regions (pixels)
        min_region_width (int): minimum width of a detection region (pixels)

    Returns:
        regions (numpy array): contains subarrays with start and end wavelengths
    """
    print('Computing detection regions...')

    N_sigma = 4.0  # Detection threshold

    num_pixels = len(wavelengths)
    min_pix = 1
    max_pix = num_pixels - 1

    std_min = 2  # Range of standard deviations for Gaussian convolution
    std_max = 11

    flux_ews = [0] * num_pixels
    noise_ews = [0] * num_pixels
    det_ratio = [-float('inf')] * num_pixels

    # Convolve varying-width Gaussians with equivalent width of flux and noise
    for std in range(std_min, std_max):

        for i in range(min_pix, max_pix):
            flux_dec = 1.0 - fluxes[i]
            if flux_dec < noise[i]:
                flux_dec = 0.0
            flux_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * flux_dec
            noise_ews[i] = 0.5 * abs(wavelengths[i - 1] - wavelengths[i + 1]) * noise[i]

        flux_ews[0] = 0
        flux_ews[max_pix] = 0
        noise_ews[0] = 0
        noise_ews[max_pix] = 0

        xarr = np.array([p - (num_pixels-1)/2.0 for p in range(num_pixels)])
        gaussian = VPfit.GaussFunction(xarr, 1.0, 0.0, std)

        flux_func = np.convolve(flux_ews, gaussian, 'same')
        noise_func = np.convolve(np.square(noise_ews), np.square(gaussian), 'same')

        # Select highest detection ratio of the Gaussians
        for i in range(min_pix, max_pix):
            noise_func[i] = 1.0 / np.sqrt(noise_func[i])
            if flux_func[i] * noise_func[i] > det_ratio[i]:
                det_ratio[i] = flux_func[i] * noise_func[i]

    pixels = [p for p in range(len(wavelengths))]

    # Select regions based on detection ratio at each point, combining nearby regions
    start = 0
    region_endpoints = []
    for i in range(len(pixels)):
        if start == 0 and det_ratio[i] > N_sigma and fluxes[i] < 1.0:
            start = pixels[i]
        elif start != 0 and (det_ratio[i] < N_sigma or fluxes[i] > 1.0):
            if pixels[i] - start > min_region_width:
                end = pixels[i]
                region_endpoints.append([start, end])
            start = 0

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

    # Combine overlapping regions and check for detection based on noise value
    regions = []
    for i in range(len(regions_expanded)):
        start = regions_expanded[i][0]
        end = regions_expanded[i][1]
        if i<(len(regions_expanded)-1) and end > regions_expanded[i+1][0]:
            end = regions_expanded[i+1][1]
        for j in range(start, end):
            flux_dec = 1.0 - fluxes[j]
            if flux_dec > abs(noise[j]) * N_sigma:
                regions.append([wavelengths[start], wavelengths[end]])
                break

    return np.array(regions)




def mock_absorption(wavelength_start=5010, wavelength_end=5030, n=3, plot=True, onesigmaerror = 0.02, saturated=False):
    """
    Generate a mock absorption profile.

    Args:
        wavelength_start (int): start of wavelength range
        wavelength_end (int): end of wavelength range
        n (int): number of absorption features
        plot (bool): plots the profile
        onesigmaerror (float): noise on profile plot

    Returns:
        clouds (pandas dataframe): dataframe containing parameters for each absorption feature
        wavelength_array (numpy array)
    """

    vpfit = VPfit()

    wavelength_array = np.arange(wavelength_start, wavelength_end, 0.01)

    clouds = pd.DataFrame({'cloud': pd.Series([], dtype='str'),
                       'amplitude': pd.Series([], dtype='float'),
                       'centroid': pd.Series([], dtype='float'),
                       'sigma': pd.Series([], dtype='float'),
                       'tau': pd.Series([], dtype='object')})

    for cloud in range(n):

        if(saturated):
            max_amplitude = 5
        else:
            max_amplitude = 1

        clouds = clouds.append({'cloud': cloud, 'amplitude': random.uniform(0, max_amplitude),
                                'centroid': random.uniform(wavelength_start+2, wavelength_end-2),
                                'sigma': random.uniform(0,2), 'tau':[]}, ignore_index=True)

        clouds.set_value(cloud, 'tau', VPfit.GaussFunction(wavelength_array, clouds.ix[cloud]['amplitude'],
                                         clouds.ix[cloud]['centroid'], clouds.ix[cloud]['sigma']))


    if plot:
        noise = np.random.normal(0.0, onesigmaerror, len(wavelength_array))
        flux_array = vpfit.Absorption(sum(clouds['tau'])) + noise

        f, (ax1, ax2) = pylab.subplots(2, sharex=True, sharey=False)

        ax1.plot(wavelength_array, sum(clouds['tau']), color='black', label='combined', lw=2)
        for c in range(len(clouds)):
            ax1.plot(wavelength_array, clouds.ix[c]['tau'], label='comp_1')

        ax2.plot(wavelength_array, flux_array, color='black')
        f.subplots_adjust(hspace=0)

        ax1.set_ylabel("Optical Depth")

        ax2.set_ylabel("Normalized Flux")
        ax2.set_xlabel("$\lambda (\AA)$")
        ax2.set_ylim(0, 1.1)

        pylab.show()

    return clouds, wavelength_array


if __name__ == "__main__":

    vpfit = VPfit()

    clouds, wavelength_array = mock_absorption(n=2)

    onesigmaerror = 0.02
    noise = np.random.normal(0.0, onesigmaerror, len(wavelength_array))
    flux_array = vpfit.Absorption(sum(clouds['tau'])) + noise

    vpfit.initialise_model(wavelength_array, flux_array, 2)
    vpfit.map_estimate()
    vpfit.mcmc_fit()

    vpfit.plot(wavelength_array, flux_array, clouds, n=2)
