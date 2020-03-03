import numpy as np

import multiprocessing as mp
import os
import h5py

from vpfits import VPfit
from vpregion import VPregion
from vpspectrum import VPspectrum


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