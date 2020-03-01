import random
import datetime

import numpy as np
import pandas as pd

import pymc as mc

import matplotlib.pyplot as plt

import multiprocessing as mp
import os
import h5py

from physics import *

from vpfits import VPfit
from vpregion import VPregion


class VPspectrum():

    def __init__(self, line, spectrum_file=None, out_folder=None, voigt=False, chi_limit=1.5, mcmc_cov=False, get_mcmc_err=True, convergence_attempts=10, 
                    chi_sq_maximum=10., max_single_region_components=15, ideal_single_region_components=5, min_region_percentage=2.):
        self.line = line
        self.spectrum_file = spectrum_file
        self.out_folder = out_folder
        self.voigt = voigt
        self.chi_limit = chi_limit
        self.mcmc_cov = mcmc_cov
        self.get_mcmc_err = get_mcmc_err
        self.convergence_attempts = convergence_attempts
        self.chi_sq_maximum = chi_sq_maximum
        self.max_single_region_components = max_single_region_components
        self.ideal_single_region_components = ideal_single_region_components
        self.min_region_percentage = min_region_percentage

        """
        Args:
            line (float): the rest wavelength of the absorption line in Angstroms
            spectrum_file (str): file with spectrum data
            out_folder (string): if plotting the fits and saving them, provide a directory. Default: None.
            voigt (boolean): switch to fit Voigt profile instead of Gaussian. Default: True.
            chi_limit (float): minimum reduced chi squared for an acceptable fit
            mcmc_cov (boolean): switch to do error propogation from the mcmc chain. Default: False
            get_mcmc_err (boolean): switch to find the standard deviation (of b and N) from the mcmc chain. Default: True
            convergence_attempts (int): number of attempts to find desirable fit
            chi_sq_maximum (float): the maximum "acceptable" chi-squared value before the line adder forcer kicks in
            max_single_region_components (int): the maximum number of regions before we impose grouping regions
            ideal_single_region_components (int): the number of regions to group into a single region if max_single_region_components is reached
            min_region_percentage (float): minimum % of pixels in a spectra a region must take up (when forcing to split regions)
        """

        if not spectrum_file is None:
            self.read_from_spectrum_file()


    def read_from_spectrum_file(self):

        with h5py.File(self.spectrum_file, 'r') as data:
            self.wavelength_array = np.array(data['wavelength'][:])
            self.noise_array = np.array(data['noise'][:])
            self.flux_array = np.array(data['flux'][:])
            self.frequency_array = Wave2freq(self.wavelength_array)


    def compute_detection_regions(self, min_region_width=2, N_sigma=4.0, extend=False, std_min=2, std_max=11):
        """
        Finds detection regions above some detection threshold and minimum width.

        Args:
            min_region_width (int): minimum width of a detection region (pixels)
            N_sigma (float): detection threshold (std deviations)
            extend (boolean): default is False. Option to extend detected regions untill tau
                            returns to continuum.
            std_min, std_max (int): range of standard deviations for Gaussian convolution
        """

        print('Computing detection regions...')

        self.num_pixels = len(self.wavelength_array)
        pixels = range(self.num_pixels)
        min_pix = 1
        max_pix = self.num_pixels - 1

        flux_ews = [0.] * self.num_pixels
        noise_ews = [0.] * self.num_pixels
        det_ratio = [-float('inf')] * self.num_pixels

        # flux_ews has units of wavelength since flux is normalised. so we can use it for optical depth space
        for i in range(min_pix, max_pix):
            self.flux_dec = 1.0 - self.flux_array[i]
            if self.flux_dec < self.noise_array[i]:
                self.flux_dec = 0.0
            flux_ews[i] = 0.5 * abs(self.wavelength_array[i - 1] - self.wavelength_array[i + 1]) * self.flux_dec
            noise_ews[i] = 0.5 * abs(self.wavelength_array[i - 1] - self.wavelength_array[i + 1]) * self.noise_array[i]

        # dev: no need to set end values = 0. since loop does not set end values
        flux_ews[0] = 0.
        noise_ews[0] = 0.

        # Convolve varying-width Gaussians with equivalent width of flux and noise
        xarr = np.array([p - (self.num_pixels-1)/2.0 for p in range(self.num_pixels)])
        
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
        for i in range(self.num_pixels):
            if start == 0 and det_ratio[i] > N_sigma and self.flux_array[i] < 1.0:
                start = i
            elif start != 0 and (det_ratio[i] < N_sigma or self.flux_array[i] > 1.0):
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
                while i > 0 and self.flux_array[i] < 1.0:
                    i -= 1
                start_new = i
                end = reg[1]
                j = end
                while j < (len(self.flux_array)-1) and self.flux_array[j] < 1.0:
                    j += 1
                end_new = j
                regions_expanded.append([start_new, end_new])

        else: regions_expanded = region_endpoints

        # Change to return the region indices
        # Combine overlapping regions, check for detection based on noise value
        # and extend each region again by a buffer
        self.region_waves = []
        self.region_pixels = []
        buffer = 3
        for i in range(len(regions_expanded)):
            start = regions_expanded[i][0]
            end = regions_expanded[i][1]
            #TODO: this part seems to merge regions if they overlap - try printing this out to see if it can be modified to not merge regions?
            if i<(len(regions_expanded)-1) and end > regions_expanded[i+1][0]:
                end = regions_expanded[i+1][1]
            for j in range(start, end):
                self.flux_dec = 1.0 - self.flux_array[j]
                if self.flux_dec > abs(self.noise_array[j]) * N_sigma:
                    if start >= buffer:
                        start -= buffer
                    if end < len(self.wavelength_array) - buffer:
                        end += buffer
                    self.region_waves.append([self.wavelength_array[start], self.wavelength_array[end]])
                    self.region_pixels.append([start, end])
                    break

        print('Found {} detection regions.'.format(len(self.region_pixels)))


    def split_difficult_region(self):

        self.difficult_fit = False #default is that the fit is fine
        # check if the fit is going to be "difficult", i.e. 1 region with way too many components
        if (len(self.region_pixels) == 1):
            start = self.region_pixels[0][0]
            end = self.region_pixels[0][1]

            fluxes = np.flip(self.flux_array[start:end], 0)
            freq = np.flip(self.frequency_array[start:end], 0)
            noise = np.flip(self.noise_array[start:end], 0)
            
            region = VPregion(fluxes, freq, noise, voigt=False, chi_limit=1.5)
            region.estimate_n()
            
            if (region.n > self.max_single_region_components):
                self.difficult_fit = True  # flag the fit as being difficult
                print(str(region.n) + " components should be in more than 1 region!")
                #TODO: find some handling for (a) damped absorbers (b) spectra which need flux to be rescaled
                region.forced_number_regions = region.n // self.ideal_single_region_components
                print("trying to force-split into " + str(forced_number_regions) + " regions.")
                ind = np.argpartition(region.flux_array, -10*region.forced_number_regions)[-10*region.forced_number_regions:]
                region.ind_sorted = np.flip(ind[np.argsort(region.flux_array[ind])], axis=0) #indicies sorted from highest flux to lowest flux
                print(str(len(region.ind_sorted)) + " possible split points to choose from")

                print("There are: " + str(region.num_pixels) + " pixels.")
                region.min_region_size = region.num_pixels * (self.min_region_percentage / 100.0)
                print("Minimum region pixels: " + str(region.min_region_size))

                region.splitting_points = [start, end]
                # original "start" is the beginning of the 1st region
                # original "end" is the end of the last region
                # each region will be contiguous, so need (forced_number_regions - 1) indexes
                # these indexes will have to be at least <min_region_size> away from each other.

                for i in range(len(region.ind_sorted)):
                    # go through indices of maximum flux, in descending order
                    # see if they can be the required distance away from each other.
                    # if they can't be made to work, try working down the list of maximum fluxes
                    if (len(region.splitting_points) == (region.forced_number_regions+1)):
                        print("Found enough splitting regions")
                        break #stop once enough points to split have been found
                    else:
                        region.dist_is_fine = True
                        for j in range(len(region.splitting_points)):
                            # check the distance between a possible "splitting point" and the existing splitting points
                            dist = abs(region.ind_sorted[i] - region.splitting_points[j])
                            if (dist < min_region_size):
                                region.dist_is_fine = False

                        if region.dist_is_fine: #if the region would be large enough, then split along this point
                            region.splitting_points.append(region.ind_sorted[i])

                print("Have managed to split into " + str(len(region.splitting_points)-1) + " regions!")
                #now make the start, end points
                region.splitting_points.sort() #sort the pixels into ascending order
                self.region_pixels = []
                self.region_waves = []

                for i in range(len(region.splitting_points)-1):
                    start = region.splitting_points[i]
                    end = region.splitting_points[i+1]
                    self.region_pixels.append([start,end]) #save the pixel numbers
                    self.region_waves.append([self.wavelength_array[start], self.wavelength_array[end]]) #save the wavelengths

    def fit_spectrum(self):
        """
        The main function. Takes an input spectrum file, splits it into manageable regions, and fits 
        the individual regions using PyMC. Finally calculates the Doppler parameter b, the 
        column density N and the equivalent width and the centroids of the absorption lines.


        Returns:
            params (dict)
        """

        # identify regions to fit in the spectrum
        self.compute_detection_regions(min_region_width=2)

        self.split_difficult_region()

        # dicts to store results
        self.params = {'b': np.array([]), 'b_std': np.array([]), 'N': np.array([]), 'N_std': np.array([]),
                    'EW': np.array([]), 'centers': np.array([]), 'region_numbers': np.array([])}

        self.flux_model = {'total': np.ones(len(self.flux_array)), 'chi_squared': np.zeros(len(self.region_pixels)), 'region_pixels': self.region_pixels,
                    'amplitude': np.array([]), 'sigmas': np.array([]), 'centers': np.array([]), 'region_numbers': np.array([]),
                    'EW': np.zeros(len(self.region_pixels)), 'std_a': np.array([]), 'std_s': np.array([]), 'std_c': np.array([]),
                    'cov_as': np.array([]), 'difficult_fit': self.difficult_fit}

        j = 0

        # TODO: move this to region class?
        self.regions = []

        for start, end in self.region_pixels:
            fluxes = np.flip(self.flux_array[start:end], 0)
            noise = np.flip(self.noise_array[start:end], 0)
            waves = np.flip(self.wavelength_array[start:end], 0)
            nu = np.flip(self.frequency_array[start:end], 0)

            region = VPregion(nu, fluxes, noise, voigt=self.voigt, chi_limit=self.chi_limit)

            region.best_chi_squared = -1 #initialize best_chi_sq

            #track the chi-squared values and associated number of components, to force-add components if necessary
            region.attempt_n = []
            region.attempt_chi_squareds = []
            
            if (region.n > self.max_single_region_components): #if there are too many components: reduce the number of attempts, and flag this spectra as difficult
                self.num_attempts = 2
                self.flux_model['difficult_fit'] = True
                if (region.n > (1.5 *self.max_single_region_components)): #fits for that many components are going to be garbage no matter how many times they're ran
                    self.num_attempts = 1
                #params['difficult_fit'] = True
            else: #otherwise, use the default setting
                self.num_attempts = self.convergence_attempts


            for _ in range(self.num_attempts):

                # make initial guess for number of lines in a region
                region.estimate_n()

                # force the fitter to add additional components if it has previously failed with fewer components
                # (even if BIC indicates otherwise)

                region.n_is_sensible = False
                while(region.n_is_sensible == False):
                    if (region.attempt_n.count(region.n) > 2): #check if it's failed to converge for a few attempts with this number of components
                        #check what the chi-squareds were for this number
                        ii = [i for i,val in enumerate(attempt_n) if val==region.n]
                        #print(ii)
                        #print(attempt_n)
                        #print(attempt_chi_squareds)
                        chi_squareds = np.array(region.attempt_chi_squareds)[ii]
                        if (np.min(chi_squareds) > self.chi_sq_maximum): #if the best chi squared value is still too high
                            print("Chi-squareds have been too high for fits using n=" + str(region.n) + " components. (chi-sq>" + str(self.chi_sq_maximum) + ")")
                            n += 1 #force it to add another component
                            print("Forcing the fitter to increase the fit to at least n=" + str(region.n) + " components.")
                        else:
                            region.n_is_sensible = True
                    else:
                        region.n_is_sensible = True
                
                # fit the region by minimising BIC and chi squared
                if (region.n > self.max_single_region_components): #less strict convergence criteria for un-splittable regions
                    print("increasing chi_limit to " + str(region.chi_limit*3) + ", because n = " + str(region.n))
                    region.iteration_chi_limit = region.chi_limit*3
                else:
                    region.iteration_chi_limit = region.chi_limit

                region.region_fit()

                # evaluate overall chi squared
                region.set_freedom()

                current_chi_squared = region.fit.ReducedChisquared(fluxes, region.fit.total.value, noise, region.freedom)
                region.attempt_n.append(region.n)
                region.attempt_chi_squareds.append(current_chi_squared)

                print('Reduced chi squared is {:.2f}'.format(current_chi_squared))

                #if this is the best (or first) chi-squared value seen so far, record it and save the fit
                if (current_chi_squared < region.best_chi_squared) or (region.best_chi_squared == -1) :
                    region.best_chi_squared = current_chi_squared
                    region.best_fit = region.fit

                # if chi squared is sufficiently small, stop there. If not, repeat the region fitting
                if region.best_chi_squared < region.iteration_chi_limit:
                    break

            # use the best fit found in the loop above
            region.fit = region.best_fit
            region.n = len(region.fit.estimated_profiles)
            self.flux_model['chi_squared'][j] = region.best_chi_squared
            print("Using model with best chi-squared seen, which is : {:.2f}".format(region.best_chi_squared))


            print('\n')
            self.flux_model['total'][start:end] = np.flip(region.fit.total.value, 0)

            self.flux_model['region_'+str(j)+'_wave'] = np.flip(waves, 0)
            self.flux_model['region_'+str(j)+'_flux'] = np.ones((region.n, len(fluxes)))
            for k in range(region.n):
                self.flux_model['region_'+str(j)+'_flux'][k] = np.flip(Tau2flux(region.fit.estimated_profiles[k].value), 0)

            self.flux_model['EW'][j] = EquivalentWidthFlux(edges=self.flux_model['region_'+str(j)+'_wave'],
                                                   fluxes=self.flux_model['region_'+str(j)+'_flux'])

            heights = np.array([region.fit.estimated_variables[i]['amplitude'].value for i in range(region.n)])
            centers = np.array([Freq2wave(region.fit.estimated_variables[i]['centroid'].value) for i in range(region.n)])
            region_numbers = np.arange(region.n)

            if not region.voigt:
                sigmas = np.array([region.fit.estimated_variables[i]['sigma'].value for i in range(region.n)])
            
            elif region.voigt:
                g_fwhms = np.array([region.fit.estimated_variables[i]['G_fwhm'].value for i in range(region.n)])
                sigmas = VPfit.GaussianWidth(g_fwhms)
            
            self.flux_model['amplitude'] = np.append(self.flux_model['amplitude'], heights)
            self.flux_model['centers'] = np.append(self.flux_model['centers'], centers)
            self.flux_model['region_numbers'] = np.append(self.flux_model['region_numbers'], region_numbers)
            self.flux_model['sigmas'] = np.append(self.flux_model['sigmas'], sigmas)

            if self.mcmc_cov:

                cov = region.fit.chain_covariance(region.n, voigt=region.voigt)
                std_a = np.sqrt([cov[i][0][0] for i in range(region.n)])
                std_s = np.sqrt([cov[i][1][1] for i in range(region.n)])
                std_c = np.sqrt([cov[i][2][2] for i in range(region.n)])
                cov_as = np.array([cov[i][0][1] for i in range(region.n)])

                self.flux_model['std_a'] = np.append(self.flux_model['std_a'], std_a)
                self.flux_model['std_s'] = np.append(self.flux_model['std_s'], std_s)
                self.flux_model['std_c'] = np.append(self.flux_model['std_c'], std_c)
                self.flux_model['cov_as'] = np.append(self.flux_model['cov_as'], cov_as)

                self.params['N_std'] = np.append(self.params['N_std'], ErrorN(heights, sigmas, std_a, std_s, cov_as))
            
            elif self.get_mcmc_err:
                stats = region.fit.mcmc.stats()
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
                std_s = np.array([stats['est_sigma_'+str(i)]['standard deviation'] for i in range(region.n)]) #mcmc fitting error for "sigmas"
                std_a = np.array([stats['xexp_'+str(i)]['standard deviation'] for i in range(region.n)])      #mcmc fitting error for "amplitude" (referred to as "xexp"

                self.flux_model['std_s'] = np.append(self.flux_model['std_s'], std_s)
                self.flux_model['std_a'] = np.append(self.flux_model['std_a'], std_a)

                self.params['b_std'] = np.append(self.params['b_std'], ErrorB(std_s, self.line))
                covariance = [0] * len(std_a) # treat covariance as 0  TODO: figure out how to extract covariance (if possible) from MCMC chain.
                self.params['N_std'] = np.append(self.params['N_std'], ErrorN(amplitude=heights,sigma=sigmas,std_a=std_a, std_s=std_s, cov_as=covariance))

            self.params['b'] = np.append(self.params['b'], DopplerParameter(sigmas, self.line))
            self.params['N'] = np.append(self.params['N'], ColumnDensity(heights, sigmas))
            self.params['centers'] = np.append(self.params['centers'], centers)
            self.params['region_numbers'] = np.append(self.params['region_numbers'], region_numbers)
            for k in range(region.n):
                self.params['EW'] = np.append(self.params['EW'], EquivalentWidthTau(region.fit.estimated_profiles[k].value, [waves[0], waves[-1]]))
            
            j += 1

            self.regions.append(region)

        if self.out_folder is not None:
            name = self.spectrum_file.split('/', -1)[-1]
            name = name[:name.find('.')] #returns e.g. "spectrum_17" (without any folder, or .h5 extension)
            self.output_filename = self.out_folder + name
            if self.voigt == True:
                self.output_filename += '_voigt_'
            else:
                self.output_filename +=  '_gauss_'

            self.plot_spectrum()
            self.write_file()

    def plot_spectrum(self):
        """
        Routine to plot the fits to the data. First plot is the total fit, second is the component
        Voigt profiles, third is the residuals.

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

        model = self.flux_model['total']

        N = 6
        length = len(self.flux_array) / N

        fig, ax = plt.subplots(N, figsize=(15,15))
        for n in range(N):
            lower_lim = int(n*length)
            upper_lim = int(n*length+length)

            ax[n].plot(self.wavelength_array, self.flux_array, c='black', label='Measured')
            ax[n].plot(self.wavelength_array, model, c='green', label='Fit')
            ax[n].set_xlim(self.wavelength_array[lower_lim], self.wavelength_array[upper_lim])
            for (start, end) in self.region_pixels:
                plot_bracket(self.wavelength_array[start], ax[n], 'left')
                plot_bracket(self.wavelength_array[end], ax[n], 'right')

        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux')
        plt.savefig(self.output_filename+'fit.png')
        plt.clf()

        fig, ax = plt.subplots(N, figsize=(15,15))
        for n in range(N):
            lower_lim = int(n*length)
            upper_lim = int(n*length+length)

            for i in range(len(self.region_pixels)):
                start, end = self.region_pixels[i]
                region_data = self.flux_model['region_'+str(i)+'_flux']
                for j in range(len(region_data)):
                    ax[n].plot(self.wavelength_array[start:end], region_data[j], c='green')
                plot_bracket(self.wavelength_array[start], ax[n], 'left')
                plot_bracket(self.wavelength_array[end], ax[n], 'right')

            ax[n].set_xlim(self.wavelength_array[lower_lim], self.wavelength_array[upper_lim])        

        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux')
        plt.savefig(self.output_filename+'components.png')
        plt.clf()    

        fig, ax = plt.subplots(N, figsize=(15,15))
        for n in range(N):
            lower_lim = int(n*length)
            upper_lim = int(n*length+length)

            ax[n].plot(self.wavelength_array, (self.flux_array - model), c='blue')
            ax[n].hlines(1, self.wavelength_array[0], self.wavelength_array[-1], color='red', linestyles='-')
            ax[n].hlines(-1, self.wavelength_array[0], self.wavelength_array[-1], color='red', linestyles='-')
            ax[n].hlines(3, self.wavelength_array[0], self.wavelength_array[-1], color='red', linestyles='--')
            ax[n].hlines(-3, self.wavelength_array[0], self.wavelength_array[-1], color='red', linestyles='--')
            ax[n].set_xlim(self.wavelength_array[lower_lim], self.wavelength_array[upper_lim])
            for (start, end) in self.region_pixels:
                plot_bracket(self.wavelength_array[start], ax[n], 'left')
                plot_bracket(self.wavelength_array[end], ax[n], 'right')

        plt.xlabel('Wavelength (A)')
        plt.ylabel('Flux')
        plt.savefig(self.output_filename+'residuals.png')
        plt.clf()

        return

    def write_file(self):
        """
        Save file with physical parameters from fit
        """
        with h5py.File(self.output_filename+'params.h5', 'a') as f:
            for p in self.params.keys():
                f.create_dataset(p, data=np.array(self.params[p]))

        with h5py.File(self.output_filename+'flux_model.h5', 'a') as f:
            for p in self.flux_model.keys():
                f.create_dataset(p, data=np.array(self.flux_model[p]))


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
    parser.add_argument('--parallel',
                        help='Number of processes to use (for running on a folder with multiple spectra)',
                        default=1, type=int)
    parser.add_argument('--conv_attempts',
                        help='Number of attempts at mcmc chain convergence',
                        default=6, type=int)

    args = parser.parse_args()
    num_procs = args.parallel

    convergence_attempts = args.conv_attempts
    chi_limit = 1.5 #TODO: take these as parameters
    mcmc_cov = False
    get_mcmc_err = True

    if (num_procs == 1):
        #single spectra processing
        name = args.data_file.split('/', -1)[-1]
        name = name[:name.find('.')]

        if args.voigt == True:
            args.output_folder += name + '_voigt_'
        else:
            args.output_folder += name + '_gauss_'

        """
        import h5py
        data = h5py.File(args.data_file, 'r')

        wavelength = np.array(data['wavelength'][:])
        noise = np.array(data['noise'][:])
        flux = np.array(data['flux'][:])
        """
        vamp = VPspectrum(args.line, args.data_file, out_folder=args.output_folder, voigt=args.voigt, convergence_attempts=convergence_attempts)
        vamp.fit_spectrum()
        #write_file(params, args.output_folder+'params.h5', 'h5')
        #write_file(flux_model, args.output_folder+'flux_model.h5', 'h5')

    elif (num_procs > 1):
        #multi-processing a folder of spectra
        spectra_folder = args.data_file
        #TODO: check this folder exists
        if not(spectra_folder.endswith("/")):
            spectra_folder += "/"

        spectra_to_fit = []
        for f in os.listdir(spectra_folder):
            if (f.startswith("spectrum_") and f.endswith(".h5")): #TODO: find a way of more flexibly specifying the "spectrum_" condition 
                file_path = spectra_folder + f
                spectra_to_fit.append(file_path)

        print("Going to fit " + str(len(spectra_to_fit)) + " spectra, using " + str(num_procs) + " processes.")

        output_folder = args.output_folder
        if not (output_folder.endswith("/")):
            output_folder += "/"
        #TODO: make folder if it doesn't exist, check if any files inside if it does (then warn)

        #Make a pool of processes to handle these files
        maxtaskperchild = 5 #the number of tasks a process can perform, before exiting (and then re-spawning - point here is to clean up things)

        #print("Number of CPUs available from mp.cpu_count(): " + str(mp.cpu_count()))
        pool = mp.Pool(processes=num_procs, maxtasksperchild=maxtaskperchild) #make a pool with the specified settings

        #ACTUALLY RUN THE THING
        #spectrum_file, line, voigt = False, chi_limit = 1.5, out_folder = None, mcmc_cov = False, get_mcmc_err = True, convergence_attempts = 10
        results = [pool.apply_async(fit_spectrum,args=(spectra, args.line, args.voigt, chi_limit, output_folder,
                                        mcmc_cov, get_mcmc_err, convergence_attempts)) for spectra in spectra_to_fit ]
        #results = [pool.apply(fit_spectrum,args=(spectra, args.line, args.voigt, output_folder, convergence_attempts)) for spectra in spectra_to_fit ]
        #pool.apply_async(fit_spectrum,args=(spectra, args.line, args.voigt, output_folder, convergence_attempts)) for spectra in spectra_to_fit 
        pool.close()
        pool.join()

        """
        for spectra in spectra_to_fit:
            #params, flux_model =
            # ACTUALLY RUN THE THING
            #pool.apply(fit_spectrum,args=(spectra, args.line, voigt=args.voigt, out_folder=output_folder, convergence_attempts=convergence_attempts))
            #pool.apply(fit_spectrum,args=(spectra, args.line, args.voigt, output_folder, convergence_attempts))
            pool.apply_async(fit_spectrum,args=(spectra, args.line, args.voigt, output_folder, convergence_attempts))
        """
        # results = [pool.apply(cube, args=(x,)) for x in range(1, 7)]
        # print(results)
    else:
        print("The --parallel setting: \"" + str(num_procs) +"\" wasn't understood.")
        print("There needs to be an integer number of processes specified.")