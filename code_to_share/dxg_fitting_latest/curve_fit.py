import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
import pickle
import fiducial_parameters as fp
from scipy.interpolate import interp1d
from frb.dm import igm
from scipy import integrate
from scipy.integrate import trapezoid
import astropy.units as u
import h5py
from scipy.optimize import curve_fit
import emcee
import corner
from tqdm import tqdm
from scipy.stats import chi2

import pymaster as nmt
import healpy as hp

import mcmcfit as mf

if __name__ == "__main__":
    # Read the DMs from the FRBs selected for the cross correlation

    FRB_path = "/home/hcwang96/projects/ctb-vkaspi/hcwang96/dmxgal/final/1_FRB_catalog/cat_II_dm_overd_repeater_once_100NE2001_inclusion.h5" 
    FRB_file = h5py.File(FRB_path, 'r')
    mean_DM = FRB_file['dm_ave'][()]
    dm_select = FRB_file['dm_od'][:] + mean_DM
    
    # define redshift bin centers from galaxy data
    zs = [0.075, 0.15, 0.25, 0.35, 0.45]
    bg = [[1.04, 0, 0], [1.11, 0, 0], [1.24, 0, 0], [1.5, 0, 0], [1.97, 0, 0]]

    f = mf.cl_Dg_fit(dm_select)

    fit_path = "/home/hcwang96/disk_full_files/cat2_referee1_more_rand/45000_combined/"
    fit_files = [fit_path + f'dmxgal_lmax8000_bin14_z{zs[i]:.3f}_nsample45000_combined.h5' for i in range(len(zs))]
    plot_path = "/home/hcwang96/projects/ctb-vkaspi/hcwang96/dmxgal/final/4_Computing_power_spectrum/DESI_LIS/9_bins/"
    plot_files = [plot_path + f'dmxgal_lmax8000_bin9_z{zs[i]:.3f}_nsample1000.h5' for i in range(len(zs))]
    f.read_data(fit_files, plot_files)

    f.mcmc(theta_init = np.array([60, -1, 4000, 1.5, -1.5, 1]), nwalkers = 32, nsteps = 50000, thin = 20, burn_in = 3000, save = True)
    f.plot_models(f.DMH_bar_mc, f.cut_scale_mc, f.l_cut_mc, f.bf0_mc, f.ALPHA_mc, f.Z_STAR, mode = 'full', convolved = False, figsize=(32, 5), save = True, filename = 'test_mean_para_val_fit')

    frb_file = "/home/hcwang96/projects/ctb-vkaspi/hcwang96/dmxgal/final/1_FRB_catalog/cat_II_dm_overd_repeater_once_100NE2001_inclusion.h5" 
    gal_path = "/lustre06/project/6034496/hcwang96/dmxgal/checks/2_Galaxy_catalog/DESI_LIS/DR8/north/"
    cat_names = ["DESI_LIS_BGS_north_z_0.05_0.1.h5", "DESI_LIS_BGS_north_z_0.1_0.2.h5", "DESI_LIS_BGS_north_z_0.2_0.3.h5", "DESI_LIS_BGS_north_z_0.3_0.4.h5", "DESI_LIS_BGS_north_z_0.4_0.5.h5"]
    ran_names = ["DESI_LIS_BGS_north_ran_all_z_v2.h5", "DESI_LIS_BGS_north_ran_all_z_v2.h5", "DESI_LIS_BGS_north_ran_all_z_v2.h5", "DESI_LIS_BGS_north_ran_all_z_v2.h5", "DESI_LIS_BGS_north_ran_all_z_v2.h5"]
    cat_files = [gal_path + cat_names[i] for i in range(len(zs))]
    ran_files = [gal_path + ran_names[i] for i in range(len(zs))]
    f.bandpower_setup(frb_file, cat_files, ran_files)
    f.plot_models(f.DMH_bar_mc, f.cut_scale_mc, f.l_cut_mc, f.bf0_mc, f.ALPHA_mc, f.Z_STAR, mode = 'full', convolved = True, figsize=(32, 5), save = True, filename = 'test_mean_para_val_fit_convolve')

    #f.bayes_factor(npoints = 30, nwalkers = 32, nsteps = 5000, burn_in = 2000, thin = 20, save = True, filename = 'bayes_factor.h5')
