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
import nestle
import dynesty
from dynesty import NestedSampler
from scipy.stats import chi2

import pymaster as nmt
import healpy as hp

from scipy.special import erf
from scipy.integrate import quad

import argparse
import mcmcfit_clfg as mf

if __name__ == "__main__":

    ps_file = '/lustre06/project/6034496/hcwang96/frbxgal/CAT_2/first_correlation/dm_bins/10_bins/frbxgal_lmax2771_12_lb_5_zb_10_db_100_samplessecond_try.h5'
    
    d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268, 444.52819007, 526.88686278, 620.77482314, 732.37124655, 886.70586238, 1155.88351684, 3916.20665172])
    Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

    g = mf.cl_fg_fit(d_bin_edge, Nd)
    g.read_data(ps_file)
    g.mcmc(save = True, thin = 10, burn_in = 1000, nsteps = 25000)