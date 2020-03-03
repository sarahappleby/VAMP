# VAMP
Voigt Automatic MCMC Profiles

## Requirements

All requirements are contained within the `requirements.txt` file. To install with pip run `pip install -r requirements.txt'.


## Usage

`python vpfits.py <pygad spectrum file> <rest wavelength>`

## Demo

Run the following in the terminal to generate a mock profile and fit with MCMC.

`python vpfits.py data/simba_CII1036.h5 1036`

A demo analysis of a synthetic spectrum from the Simba simulations in contained within `simba_spec_demo.ipynb`. `quasar_spec_demo.ipynb` contains a similar analysis of the quasar Q1422+231. A look at the main class `VPfit` is contained within `vpfits_intro.ipynb`. You need Jupyter to run these notebooks. Run the following in the same directory as the notebook to start the jupyter server.

`jupyter notebook`

