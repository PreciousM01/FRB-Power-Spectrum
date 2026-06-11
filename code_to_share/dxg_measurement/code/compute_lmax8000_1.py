import argparse
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import camb
from camb import model, initialpower
import pickle
from scipy.interpolate import interp1d
from frb.dm import igm
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

import os
from astropy.io import fits

class dmxgal:
    def __init__(self, FRB_path, gal_path, cat_names, ran_names, zbin_l = [0.05, 0.1, 0.2], zbin_r = [0.1, 0.2, 0.4], lmax = 8000, bin_edges = [1, 10, 20, 30, 40, 67, 115, 196, 333, 565, 960, 1632, 2772, 4709, 8001]):
        self.FRB_path = FRB_path
        self.gal_path = gal_path
        self.cat_names = cat_names
        self.ran_names = ran_names
        assert len(self.cat_names) == len(self.ran_names)
        self.ncat = len(self.cat_names)
        
        self.zbin_l = zbin_l
        self.zbin_r = zbin_r
        assert len(self.zbin_l) == len(self.zbin_r)
        assert len(self.zbin_l) == len(self.cat_names)
        if len(self.zbin_l) == 3:
            self.zbin_notes = 'zall'
        if len(self.zbin_l) == 1:
            zbin_center = (self.zbin_l[0] + self.zbin_r[0])/2
            self.zbin_notes = f'z{zbin_center:.3f}'

        self.lmax = lmax
        self.bin_edges = bin_edges
        if self.bin_edges[-1] != self.lmax+1:
            self.bin_edges[-1] = self.lmax+1
        self.bin = nmt.NmtBin.from_edges(self.bin_edges[:-1], self.bin_edges[1:])
        self.leff = self.bin.get_effective_ells()
        self.bin_num = len(self.leff)

        self.ps = self.measure_ps()

    def read_frb_cat(self, path):
        file = h5py.File(path, 'r')
        l_deg = file["l_deg"][:]
        b_deg = file["b_deg"][:]
        dm_od = file["dm_od"][:]
        w = np.ones_like(l_deg)
        return [l_deg, b_deg], w, dm_od
    
    def read_h5_cat(self, path):
        file = h5py.File(path, 'r')
        l_deg = file["l_deg"][:]
        b_deg = file["b_deg"][:]
        try: 
            w = file['w'][:]
        except KeyError:
            w = np.ones_like(l_deg)
        return [l_deg, b_deg], w

    def measure_ps(self):
        
        ps = np.zeros((self.ncat, self.bin_num))
        
        # Not implementing the theory template yet
        #self.theory_cl = np.zeros((self.ncat, self.bin_num))
        #ls = np.arange(self.lmax + 1) # we need from 0 to lmax

        self.frb_pos, self.frb_w, self.frb_dmo = self.read_frb_cat(self.FRB_path)
        self.gal_clu = []
        self.cat_w = []

        for i, (cat_name, ran_name) in enumerate(zip(self.cat_names, self.ran_names)):
            cat_path = self.gal_path + cat_name
            ran_path = self.gal_path + ran_name
            cat_pos, cat_w = self.read_h5_cat(cat_path)
            ran_pos, ran_w = self.read_h5_cat(ran_path)

            #print(f"Estimating power spectrum for redshift bin {i}")
        
            # compute the DM-galaxt cross power spectrum using the catalog based estimator
            frb_cat = nmt.NmtFieldCatalog(self.frb_pos, self.frb_w, self.frb_dmo, lmax=self.lmax, lonlat=True)
            gal_clu = nmt.NmtFieldCatalogClustering(cat_pos, cat_w, ran_pos, ran_w, lmax=self.lmax, lonlat=True)
            cat_w = nmt.NmtWorkspace.from_fields(frb_cat, gal_clu, self.bin)
            pcl = nmt.compute_coupled_cell(frb_cat, gal_clu)
            cat_cl = cat_w.decouple_cell(pcl)

            ps[i][:] = cat_cl[0]

            self.gal_clu.append(gal_clu)
            self.cat_w.append(cat_w)

            # get convolved theory prediction, not implemented yet
            #temp = get_theory_temp([self.zbin_l[i]], [self.zbin_r[i]], [self.bg[i]], ls)
            #self.theory_cl[i] = cat_w.decouple_cell(cat_w.couple_cell(temp.cl))[0]

        return ps

    def null_test(self, nsample):
        self.nsample = nsample
        self.rand_ps = np.zeros((self.ncat, self.bin_num, self.nsample))
        
        for zbin in range(self.ncat):

            #print(f"Estimating error bar for redshift bin {zbin}")
        
            for i in range(self.nsample):
                
                print(i)
                # randomly permutating FRB DMs while keeping their positions fixed
                f_cat_rand = nmt.NmtFieldCatalog(self.frb_pos, self.frb_w, np.random.permutation(self.frb_dmo), lmax=self.lmax, lonlat=True)
                pcl_rand = nmt.compute_coupled_cell(f_cat_rand, self.gal_clu[zbin])
                
                # no need to recompute the mode coulping matrix since it depends on the masks only which do not change
                #w_fg_rand = nmt.NmtWorkspace.from_fields(f_cat_rand, g_clus, b)
                
                cl_fg_rand = self.cat_w[zbin].decouple_cell(pcl_rand)
                self.rand_ps[zbin, :, i] = cl_fg_rand[0][:]

        self.null_ps_mean = np.mean(self.rand_ps, axis=-1)
        self.error_bar = np.std(self.rand_ps, axis=-1)
        self.null_ps_mean_error = self.error_bar/np.sqrt(self.nsample) # standard deviation of the mean

    def save(self, notes =''):

        outname = f"dmxgal_lmax{self.lmax}_bin{self.bin_num}_{self.zbin_notes}_nsample{self.nsample}{notes}.h5"
        file = h5py.File(outname, 'w')
        file.create_dataset('lmax', data = self.lmax)
        file.create_dataset('bin_edges', data = self.bin_edges)
        file.create_dataset('bin_centers', data = self.leff)
        file.create_dataset('zbin_l', data = self.zbin_l)
        file.create_dataset('zbin_r', data = self.zbin_r)
        file.create_dataset('cl', data = self.ps)
        file.create_dataset('error', data = self.error_bar)
        file.create_dataset('null_cl_mean', data = self.null_ps_mean)
        file.create_dataset('null_cl_error', data = self.null_ps_mean_error)
        file.create_dataset('rand_ps', data = self.rand_ps)
        file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the DM-galaxy cross power spectrum"
    )
    
    parser.add_argument("--nsample", required=False, type=int)
    args = parser.parse_args()

    FRB_path = '/home/hcwang96/projects/ctb-vkaspi/hcwang96/dmxgal/final/1_FRB_catalog/cat_II_dm_overd_repeater_once_100NE2001_inclusion.h5'
    gal_path = '/lustre06/project/6034496/hcwang96/dmxgal/checks/2_Galaxy_catalog/DESI_LIS/DR8/north/'
    cat_names = ["DESI_LIS_BGS_north_z_0.05_0.1.h5"]
    ran_names = ["DESI_LIS_BGS_north_ran_all_z_v2.h5"]
    zbin_l = [0.05]
    zbin_r = [0.1]
    lmax = 8000
    bin_edges = [1, 10, 20, 30, 40, 67, 115, 196, 333, 565, 960, 1632, 2772, 4709, 8001]
    nsample = args.nsample

    f = dmxgal(FRB_path, gal_path, cat_names, ran_names, zbin_l, zbin_r, lmax, bin_edges)
    f.null_test(nsample)
    f.save()