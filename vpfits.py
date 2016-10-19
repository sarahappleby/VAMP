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

    def GaussFunction(self, wavelength_array, amplitude, centroid, sigma):
        """
        Gaussian.

        Args:
            wavelength_array (numpy array)
            amplitude (float)
            centroid (float): must be between the limits of wavelength_array
            sigma (float)
        """
        return amplitude * np.exp(-0.5 * ((wavelength_array - centroid) / sigma)**2)

    def Absorption(self, arr):
        """
        Convert optical depth in to absoprtion profile

        Args:
            arr (numpy array): array of optical depth values
        """

        return np.exp(-arr)

    def initialise_components(self, wavelength_array, n, sigma_max = 5):
        """
        Initialise each fitted component of the model. Each component consists of three variables, height, centroid and sigma. These variables are encapsulated in a deterministic profile variable. The variables are stored in a dictionary, `estimated_variables`, and the profiles in a list, `estimated_profiles`.

        Args:
            wavelength_array (numpy array)
            n (int): number of components
            sigma_max (float): maximum permitted range of fitted sigma values
        """

        self.estimated_variables = {}
        self.estimated_profiles = []

        for component in range(n):
            self.estimated_variables[component] = {}

            self.estimated_variables[component]['height'] = mc.Uniform("est_height_" + str(component), 0, 1)

            self.estimated_variables[component]['centroid'] = mc.Uniform("est_centroid_" + str(component),
                                                                         wavelength_array[0], wavelength_array[-1])

            self.estimated_variables[component]['sigma'] = mc.Uniform("est_sigma_" + str(component), 0, sigma_max)

            @mc.deterministic(trace = True)
            def profile(x=wavelength_array,
                        centroid=self.estimated_variables[component]['centroid'],
                        sigma=self.estimated_variables[component]['sigma'],
                        height=self.estimated_variables[component]['height']):
                return self.GaussFunction( x, height, centroid, sigma )

            self.estimated_profiles.append(profile)


    def plot(self, wavelength_array, flux_array, clouds, n, onesigmaerror = 0.02):
        """
        Plot the fitted absorption profile

        Args:
            wavelength_array (numpy array):
            flux_array (numpy array): original flux array, same length as wavelength_array
            clouds (pandas dataframe): dataframe containing details on each absorption feature
            n (int): number of *fitted* absorption profiles
            onesigmaerror (float): noise on profile plot
        """

        f, (ax1, ax2, ax3) = pylab.subplots(3, sharex=True, sharey=False, figsize=(10,10))

        ax1.plot(wavelength_array, (flux_array - self.total.value) / onesigmaerror)
        ax1.hlines(1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')
        ax1.hlines(-1, wavelength_array[0], wavelength_array[-1], color='red', linestyles='-')

        ax2.plot(wavelength_array, flux_array, color='black', linewidth=1.0)

        for c in range(len(clouds)):
            if c==0:
                ax2.plot(wavelength_array, self.Absorption(clouds.ix[c]['tau']),
                         color="red", label="Actual", lw=1.5)
            else:
                ax2.plot(wavelength_array, self.Absorption(clouds.ix[c]['tau']),
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



    def fit(self, wavelength, flux, n):
        """
        MCMC fit of `n` absorption profiles to a given spectrum

        Args:
            wavelength (numpy array)
            flux (numpy array): flux values at each wavelength
            n (int): number of absorption profiles to fit
        """

        self.initialise_components(wavelength, n)

        # deterministic variable for the full profile
        @mc.deterministic(trace=False)
        def total(profile_sum=self.estimated_profiles):
            return self.Absorption(sum(profile_sum))

        self.total = total

        # represent full profile as a normal
        self.profile = mc.Normal("obs", self.total, self.std_deviation, value=flux, observed=True)

        # create model with parameters of all profiles to be fitted
        self.model = mc.Model([self.estimated_variables[x][y] for x in self.estimated_variables for y in self.estimated_variables[x]])# + [std_deviation])


        # Calculate the Maximum A Posteriori (MAP) estimate. Useful to do in advance so as to start the sampling with good initial values
        self.MAP = mc.MAP(self.model)
        self.MAP.fit()

        # create MCMC object
        self.mcmc = mc.MCMC(self.model)

        # fit the model
        starttime=datetime.datetime.now()
        self.mcmc.sample(iter=10000, burn=6000, thin=2.0)
        self.fit_time = str(datetime.datetime.now() - starttime)
        print "\nTook:", self.fit_time, " to finish."



def mock_absorption(wavelength_start=5010, wavelength_end=5030, n=3, plot=True, onesigmaerror = 0.02):
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
        clouds = clouds.append({'cloud': cloud, 'amplitude': random.uniform(0,1),
                                'centroid': random.uniform(wavelength_start+2, wavelength_end-2),
                                'sigma': random.uniform(0,2), 'tau':[]}, ignore_index=True)

        clouds.set_value(cloud, 'tau', vpfit.GaussFunction(wavelength_array, clouds.ix[cloud]['amplitude'],
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

    vpfit.fit(wavelength_array, flux_array, 2)

    vpfit.plot(wavelength_array, flux_array, clouds, n=2)
