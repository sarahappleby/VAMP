import numpy as np

import vpfits


def fit_profile(wavelength_array, flux_array, noise, r_threshold=0.95):

    regions = vpfits.compute_detection_regions(wavelength_array, flux_array, noise)

    fits = []
    for region in regions:
        start = np.where(wavelength_array == region[0])[0][0]
        end = np.where(wavelength_array == region[1])[0][0]
        wls = wavelength_array[start:end]
        fs = flux_array[start:end]

        r = 0
        n = 1

        vpfit_curr = vpfits.VPfit()
        vpfit_curr.fit(wls, fs, n)

        while r < r_threshold:
            vpfit_prev = vpfit_curr

            vpfit_curr = vpfits.VPfit()
            vpfit_curr.fit(wls, fs, n+1)

            r = vpfit_prev.mcmc.BPIC / vpfit_curr.mcmc.BPIC

            n += 1

        fits.append(vpfit_prev)

    return fits