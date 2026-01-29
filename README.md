# Improved Jeans model for SIDM halo profiles

An analytical model for the density profiles of self-interacting dark-matter halos with inhabitant galaxies, affiliated with the paper of Jia, et al. (2026)

- Installation

`git clone https://github.com/ZixiangJia/SIDM_Jeans_model.git`

- Model overview

This model combines the isothermal Jeans model and the model of adiabatic halo contraction into a simple semi-analytic procedure for computing the density profile of self-interacting dark-matter (SIDM) haloes with the gravitational influence from the inhabitant galaxies. 

This is an upgrated version based on the model in Jiang, et al. (2023).  Our improved model incorporates (i) velocity-dependent cross sections, (ii) an empirical treatment of gravothermal core collapse, and (iii) enhanced numerical robustness for identifying multiple solutions.

- Workflow of the model

(1) Given a CDM halo described by an NFW profile (i.e., with known virial mass, concentration, and age), and given an inhabitant galaxy described by a Hernquist profile (parameterized by the mass and scale size), compute the adiabatically contracted halo profile.

(2) Given the product of self-interaction cross-section and halo age, solve for the radius of frequent scattering, r1, using the density profile and velocity-dispersion profile of the contracted CDM halo. If the two solution branches of Jeans model have merged (product > product_merge), the product should be mirrored about product_merge, which is the product value where the two solution branches merge.

(3) Integrate the spherical Jeans-Poisson equation to obtain an isothermal core profile -- do this iteratively to find the central DM density and the central velocity dispersion by minimizing the relative stitching error at r1. If the two solution branches of Jeans model have not merged (product < product_merge, corresponding to the core-growth stage), return the low-density solution. If not, the halo has entered the core-collapse stage (product > product_merge) and the two solution branches have merged, we approximate the profile using the high-density solution at product_mirror = 2*product_merge - product. 

- Modules

profiles.py: density-profile classes (NFW, coreNFW, Hernquist, Miyamoto-Nagai, Burkert, Einasto, Dekel-Zhao, DC14, etc.), as well as the isothermal Jeans model for SIDM halos of Kaplinghat et al. (2014, 2016)

cosmo.py: cosmology-related functions

config.py: global variables and user controls 

galhalo.py: galaxy-halo connections, including the adiabatic contraction calculations of Gnedin et al. (2004, 2011)

aux.py: auxiliary functions

- Dependent libraries and packages

numpy, scipy, cosmolopy, fast_histogram, lmfit

We recommend using python installations from Enthought or Conda. 

- Functions

1. for effective cross section: compute_sigmaeff, create_sigmaeff_vmax_interpolation in profiles.py

2. for adiabatic contraction of the halo: contra_general_Minterp, r1_direct_contra in galhalo.py

3. for quantities of the halo: r1, tmerge in profiles.py

4. for profiles of SIDM isothermal cores: stitchSIDMcore_given_pmerge (suggested), stitchSIDMcore2 in profiles.py

- Example notebooks

example_profiles.ipynb -- a notebook that computes and plots the SIDM halo profiles.

exmaple_halo_evolution.ipynb -- a notebook that computes and plots the evolutionary track of SIDM halos (central density vs halo age * cross section).


