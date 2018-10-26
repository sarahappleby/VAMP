import numpy as np

constants = {'c': {'value': 2.98e8, 'units': 'm/s', 'def': 'Speed of light in a vacuum'}, 
            'sigma0': {'value': 2.36e-6, 'units': 'm**2 / s', 'def': 'Cross section for absorption'}}

def ColumnDensity(amplitude, sigma):
    """
    Find the column density of an absorption line.
    
    Args:
        amplitude (numpy array): the amplitudes of the profile fit in frequency space.
        sigma (numpy array): the std deviations of the profile fit in frequency space.

    """
    return amplitude*sigma*np.sqrt(2*np.pi) / constants['sigma0']['value']

def DopplerParameter(sigma, line):
    """
    Find the Doppler b parameter of an absorption line.

    Args:
        sigma (numpy array): the std deviation of the Gaussian in frequency space.
        line (float): the rest wavelength of the absorption line in Angstroms
    """
    # convert line position from Angstroms to m.
    return line*1.e-13*sigma / np.sqrt(2)

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
