import numpy as np

constants = {'c': {'value': 2.98e8, 'units': 'm/s', 'def': 'Speed of light in a vacuum'}, 
            'sigma0': {'value': 0.0263, 'units': 'cm**2 / s', 'def': 'Cross section for absorption'}}

def ColumnDensity(amplitude, sigma):
    """
    Find the column density of an absorption line in cm**-2.
    
    Args:
        amplitude (numpy array): the amplitudes of the profile fit in frequency space.
        sigma (numpy array): the std deviations of the profile fit in frequency space.

    """
    return amplitude*sigma*np.sqrt(2*np.pi) / constants['sigma0']['value']

def DopplerParameter(sigma, line):
    """
    Find the Doppler b parameter of an absorption line in km**-1.

    Args:
        sigma (numpy array): the std deviation of the Gaussian in frequency space.
        line (float): the rest wavelength of the absorption line in Angstroms
    """
    # convert line position from Angstroms to m.
    line *= 1.e-10
    return (line*sigma / np.sqrt(2))*1.e-3

def EquivalentWidth(taus, edges):
    """
    Find the equivalent width of a line/region.
    
    Args:
        taus (numpy array): the optical depths.
        edges (list): the edges of the regions, in either frequency or 
        wavelength space.
    Returns:
        Equivalent width in units of the edges.
    """
    return np.sum(1 - np.exp(-1*taus)) * np.abs((edges[-1] - edges[0]))

def ErrorB(std_s, line):
    """
    Evaluate the standard deviation on the Doppler b parameter from the standard
    deviation of the profile fit width sigma.
    Args:
        std_s (numpy array): the standard deviation of the width
        line (float) the rest wavelength of the absorption line in Angstroms
    """
    return DopplerParameter(std_s, line)

def ErrorN(amplitude, sigma, std_a, std_s, cov_as):
    """
    Evaluate the standard deviation on the column density N from the standard deviations
    of the profile fit parameters
    Args:
        amplitude (numpy array): the amplitudes of the profile fit in frequency space.
        sigma (numpy array): the widths of the profile fit in frequency space.
        std_a (numpy array): the standard deviations of the amplitude
        std_s (numpy array): the standard deviations of the widths
        cov_as (numpy array): the covariance of amplitude and width
    """
    prefactor = np.sqrt(2.*np.pi) / constants['sigma0']['value']
    amp_part = sigma**2 * std_a **2
    sig_part = amplitude**2 * std_s**2
    #cov_part = 2*cov_as * amplitude * sigma
    #return prefactor * np.sqrt(amp_part + sig_part + cov_part)
    return prefactor * np.sqrt(amp_part + sig_part) #currently ignoring covariance


def Errorl(std_f):
    """
    Evaluate the standard deviation on the line centroid position.
    Args:
        std_f (numpy array): the standard deviations of the line positions in frequency space
    """
    return constants['c']['value']*std_f / 1.e-10

def Tau2flux(tau):
    """
    Convert optical depth to normalised flux profile.

    Args:
        tau (numpy array): array of optical depth values
    """
    return np.exp(-tau)

def Flux2tau(flux):
    """
    Convert normalised flux to optical depth.

    Args:
        flux (numpy array): array of fluxes
    """
    return -1*np.log(flux)

def Freq2wave(frequency):
    """
    Convert frequency to wavelength in Angstroms
    """
    return (constants['c']['value'] / frequency) / 1.e-10

def Wave2freq(wavelength):
    """
    Convert wavelength in Angstroms to frequency
    """
    return constants['c']['value'] / (wavelength*1.e-10)

def Wave2red(wave, rest_wave):
    """
    Args:
        wave (numpy array): array of wavelengths
        wave_rest (float): rest wavelength
    """
    return (wave - rest_wave) / rest_wave
