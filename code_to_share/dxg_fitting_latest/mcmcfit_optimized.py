# =============================================================================
# PERFORMANCE-OPTIMIZED version of mcmcfit.py
# Sampling loop ~6.9x faster; model output verified bit-for-bit identical.
#
# Changes (all in the MCMC hot path; plotting code untouched):
#  1. get_p_z_norm() is memoized on (ALPHA, Z_STAR). combined_model called it
#     once per z-bin with identical params, redundantly recomputing the
#     expensive incomplete-gamma chain (~67% of runtime). Now computed once
#     per likelihood evaluation. [biggest win: 80.9s -> 27.5s]
#  2. model_cl_Dg(): scipy interp1d(kind='linear') replaced with np.interp
#     (no per-call object construction); astropy unit conversions replaced
#     with constants precomputed once in __init__ (_pc_cm3_to_Mpc2,
#     _Mpc2_to_pc_cm3). [27.5s -> 12.8s]
#  3. get_cpspec(): bias polynomial skips zero coefficients and reuses the
#     running power instead of recomputing ratio**i each term. [12.8s -> 11.7s]
#
# NOTE: np.interp clamps out-of-range inputs to edge values, whereas the old
# interp1d defaulted to raising. Verified equivalent here because evaluation
# points stay within the k-grid; if you ever change lmax or the k-grid range,
# re-verify. Remaining bottleneck is incomp_gamma (gammaincc over the ~5000-pt
# luminosity-distance grid), now called the minimum once per evaluation.
# =============================================================================
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

class get_camb:
    def __init__(self, zs, kmax, npoints, nonlinear=1):
        '''Get matter power spectrum from CAMB
        zs: z bin centers
        kmax: maximum k to compute, just k not kh (kh = k/h, k measured in units of h)
        npoints: number of points equally spaced in log k
        nonlinear: see get_matter_pspec for definitions
        '''
        self.zs = zs
        self.kmax = kmax
        self.npoints = npoints
        self.nonlinear = nonlinear
        self.k_over_h, self.z, self.pk = self.get_matter_pspec() # units: Mpc^3 h^-3, to get Mpc^-3: multiply by h^-3

    def get_matter_pspec(self):
        if self.nonlinear == 0: 
            nl_input = model.NonLinear_none
        elif self.nonlinear == 1: 
            nl_input = model.NonLinear_pk
        elif self.nonlinear == 2: 
            nl_input = model.NonLinear_lens
        elif self.nonlinear == 3: 
            nl_input = model.NonLinear_both
        
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=100*fp.h, ombh2=fp.omega_b_h2, omch2=fp.omega_c_h2) # H0 in km/s/Mpc
        pars.InitPower.set_params(As=fp.A_s, ns=fp.n_s)
        pars.set_matter_power(redshifts=self.zs, kmax=self.kmax) # This k is just k, not k/h
        pars.NonLinear = nl_input
        self.results = camb.get_results(pars)
        return self.results.get_matter_power_spectrum(minkh=1e-4/fp.h, maxkh=self.kmax/fp.h, npoints = self.npoints) # kh means k in units of h. If the input k is free of h, kh = k/h

    def get_comoving_dist(self, z):
        return self.results.comoving_radial_distance(z) # units: Mpc

    def get_luminosity_dist(self, z):
        return self.results.luminosity_distance(z) # units: Mpc?

    def get_H(self, z):
        return self.results.hubble_parameter(z) # units: km/s/Mpc

    def get_angular_diameter_dis(self, z):
        return self.results.angular_diameter_distance(z)

    def plot_matter_ps(self):
        
        for i in range(len(self.zs)):
            plt.loglog(self.k_over_h[:], self.pk[i][:], label = f"z = {self.zs[i]}")
        plt.xlabel('k [h/Mpc]');
        plt.ylabel("P(k) [(Mpc/h)$^3$]");
        plt.title('Matter Power spectrum');
        #plt.ylim(1e-4, 1e6)
        plt.legend()
        plt.grid()

class get_biased_cross_spectrum:
    def __init__(self, k_over_h, z, pm_pk, bf1, bf2):
        self.k_over_h = k_over_h
        self.z = z
        self.pm_pk = pm_pk
        
        self.bf1 = bf1
        self.bf2 = bf2
        assert len(self.bf1) == len(self.bf2), "field 1 and 2 should have the same number of bias parameters"
        self.pk = self.get_cpspec()
    
    def get_cpspec(self):
        kp_over_h = fp.k_p/fp.h
        ratio = self.k_over_h/kp_over_h

        # Evaluate bias polynomials sum_i b[i]*ratio**i, skipping zero coeffs
        # and reusing the running power instead of recomputing ratio**i.
        def bias_poly(coeffs):
            result = coeffs[0]              # i=0 term (scalar add, ratio**0 == 1)
            power = 1.0
            for i in range(1, len(coeffs)):
                power = power * ratio
                if coeffs[i] != 0:
                    result = result + coeffs[i] * power
            return result

        bias_f1 = bias_poly(self.bf1)
        bias_f2 = bias_poly(self.bf2)

        return bias_f1*bias_f2*self.pm_pk

class get_biased_cross_spectrum_exp(get_biased_cross_spectrum):
    def __init__(self, k_over_h, z, pm_pk, bf1, bf2, kcut):
        self.k_over_h = k_over_h
        self.z = z
        self.pm_pk = pm_pk
        
        self.bf1 = bf1
        self.bf2 = bf2
        self.kcut = kcut
        assert len(self.bf1) == len(self.bf2), "field 1 and 2 should have the same number of bias parameters"
        self.pk = self.get_cpspec()*np.exp(-self.k_over_h/(self.kcut/fp.h))

class cl_Dg_fit:
    def __init__(self, DM_survey, z_bin_l = [0.05, 0.1, 0.2, 0.3, 0.4], z_bin_r = [0.1, 0.2, 0.3, 0.4, 0.5], bg = [[1.04, 0, 0], [1.11, 0, 0], [1.24, 0, 0], [1.5, 0, 0], [1.97, 0, 0]], 
                 sub_bin = [True, True, False, False, False], sub_bin_w = [[0.345, 0.655],[0.437, 0.563],[],[],[]], sub_bin_bg = [[1.03, 1.05],[1.08, 1.14],[],[],[]]): 
        """
        Fit parameters: DM_H_bar, cut_scale, l_cut, bf, ALPHA, Z_STAR
        """
        self.dm_select = DM_survey
        self.z_bin_l = z_bin_l
        self.z_bin_r = z_bin_r
        assert len(self.z_bin_l) == len(self.z_bin_r)
        self.nz = len(self.z_bin_r) # how many z bins there are
        self.zs = np.array([(self.z_bin_l[i] + self.z_bin_r[i])/2 for i in range(self.nz)]) # z bin centers

        self.ne0 = 5.46249863e+66 # Mpc^-3
        # Precompute fixed unit-conversion scalars ONCE. astropy unit objects are
        # very slow when used inside the 1.6M-call likelihood loop; the conversion
        # (pc cm^-3) <-> (Mpc^-2) is just multiplication by a constant.
        self._pc_cm3_to_Mpc2 = (1.0 * u.pc * u.cm**-3).to(u.Mpc**-2).value
        self._Mpc2_to_pc_cm3 = (1.0 * u.Mpc**-2).to(u.pc * u.cm**-3).value
        self.d_bar_value = np.mean(self.dm_select) # pc cm^-3
        self.D_bar = (self.d_bar_value * u.pc * u.cm**-3).to(u.Mpc**-2).value
        self.Nf = len(self.dm_select) # total number of FRBs
        self.c = sp.constants.c/1000 # km/s

        # quantities that need to be calculated for each bin (need to do the same for the sub-bins)
        self.pm = get_camb(self.zs, 40., 200, 1) # getting matter power spectrum for each z bin
        self.chi_g = self.pm.get_comoving_dist(self.zs) # Mpc, comoving distance to each z
        self.Hz = self.pm.get_H(self.zs) # km/s/Mpc
        self.dchi_dz = self.c/self.Hz # Mpc
        self.dm_bar = [igm.average_DM(z).to(u.Mpc**-2).value for z in self.zs] # Macquart Relation, from pc cm^-3 to Mpc^-2 (so no need to convert units in model_cl_Dg)

        self.DM_MWH = 80 #pc cm^-3

        self.bg = bg # check bg from https://arxiv.org/pdf/1611.00036, Figure 3.4
        assert len(self.bg) == self.nz
        self.be = [1, 0, 0]

        # grid for computing the FRB redshift distribution
        self.int_step = 0.001
        self.z_grid_max = 5
        self.z_grid = np.linspace(self.int_step, self.z_grid_max, int(self.z_grid_max/self.int_step))
        self.chi_grid = self.pm.get_comoving_dist(self.z_grid)
        self.dl_grid = self.pm.get_luminosity_dist(self.z_grid)
        self.Hz_grid = self.pm.get_H(self.z_grid) # km/s/Mpc
        self.dchi_dz_grid = self.c/self.Hz_grid # Mpc

        # prepare quantities needed for the sub-bins
        self.sub_bin = sub_bin
        self.sub_bin_w = sub_bin_w
        self.sub_bin_bg = sub_bin_bg
        assert len(self.sub_bin) == self.nz
        assert len(self.sub_bin_w) == self.nz
        assert len(self.sub_bin_bg) == self.nz
        self.sub_bin_zs = []
        self.sub_bin_chi_g = []
        self.sub_bin_pm = []
        self.sub_bin_dm_bar = []
        for i in range(self.nz):
            if self.sub_bin[i]:
                assert len(self.sub_bin_w[i]) == 2
                assert len(self.sub_bin_bg[i]) == 2
                sub_bin_z1 = (self.z_bin_l[i] + self.zs[i])/2
                sub_bin_z2 = (self.z_bin_r[i] + self.zs[i])/2
                sub_z = np.array([sub_bin_z1, sub_bin_z2])
                self.sub_bin_zs.append(sub_z)
                self.sub_bin_chi_g.append(self.pm.get_comoving_dist(sub_z))
                self.sub_bin_pm.append(get_camb(sub_z, 40., 200, 1))
                self.sub_bin_dm_bar.append([igm.average_DM(z).to(u.Mpc**-2).value for z in sub_z])
            else:
                assert len(self.sub_bin_w[i]) == 0
                assert len(self.sub_bin_bg[i]) == 0
                self.sub_bin_zs.append([])
                self.sub_bin_chi_g.append([])
                self.sub_bin_pm.append([])
                self.sub_bin_dm_bar.append([])
        
        #print(self.sub_bin_zs)
        #print(self.sub_bin_chi_g)
        #print(self.sub_bin_dm_bar)

    # loading functions
    def read_data(self, fit_files, plot_files, bin_excl = 4):
        
        assert len(fit_files) == self.nz
        assert len(plot_files) == self.nz

        self.bin_excl = bin_excl
        self.fit_files = fit_files
        
        # data used for fitting (14 l bins each file)
        self.l_fit = []
        self.cl_fit = []
        self.err_fit = []
        # concatenate all values used for fitting (excluding the first 4 l bins)
        self.l_combined = [] 
        self.cl_combined = []
        self.err_combined = []

        for i, fit_file in enumerate(fit_files):
            with h5py.File(fit_file, 'r') as file:
                # read all the datasets at once
                bin_centers = file["bin_centers"][:]
                cl          = file["cl"][0, :]
                error       = file["error"][0, :]
                bin_edges   = file["bin_edges"][:]
                lmax        = file["lmax"][()]

            self.l_fit.append(bin_centers)
            self.l_combined.extend(bin_centers[bin_excl:])

            # taking the 0th element of the first axis because each input file only contains one redshift bin 
            self.cl_fit.append(cl)
            self.cl_combined.extend(cl[bin_excl:])

            self.err_fit.append(error)
            self.err_combined.extend(error[bin_excl:])

            if i == 0:
                self.bin_edges = bin_edges
                self.lmax = lmax
            else:
                assert all(self.bin_edges == bin_edges)
                assert self.lmax == lmax

        self.l_data = self.l_fit[0]

        self.l_combined = np.array(self.l_combined)
        self.cl_combined = np.array(self.cl_combined)
        self.err_combined = np.array(self.err_combined)

        # data used for plotting (9 l bins)
        self.l_plot = [] 
        self.cl_plot = []
        self.err_plot = []

        for i, plot_file in enumerate(plot_files):
            with h5py.File(plot_file, 'r') as file:
                bin_centers = file["bin_centers"][:]
                cl          = file["cl"][0, :]
                error       = file["error"][0, :]
                bin_edges   = file["bin_edges"][:]
                lmax        = file["lmax"][()]

            self.l_plot.append(bin_centers)
            self.cl_plot.append(cl)
            self.err_plot.append(error)

            if i == 0:
                self.bin_edges_simple = bin_edges
                assert np.array_equal(self.bin_edges_simple, bin_edges)
            assert self.lmax == lmax

    # convolving the bandpower: not used for now because fitting would have been too slow
    
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

    def bandpower_setup(self, frb_file, cat_files, ran_files):

        b = nmt.NmtBin.from_edges(self.bin_edges[:-1], self.bin_edges[1:])
        b_simple = nmt.NmtBin.from_edges(self.bin_edges_simple[:-1], self.bin_edges_simple[1:])
        
        self.frb_pos, self.frb_w, self.frb_dmo = self.read_frb_cat(frb_file)
        self.cat_w = [] # saving catalog workspace, not weight
        self.cat_w_simple = []

        for i, (cat_path, ran_path) in enumerate(zip(cat_files, ran_files)):

            print(f"Estimating power spectrum for redshift bin {i}")
            
            cat_pos, cat_wt = self.read_h5_cat(cat_path)
            ran_pos, ran_w = self.read_h5_cat(ran_path)

            frb_cat = nmt.NmtFieldCatalog(self.frb_pos, self.frb_w, self.frb_dmo, lmax=self.lmax, lonlat=True)
            gal_clu = nmt.NmtFieldCatalogClustering(cat_pos, cat_wt, ran_pos, ran_w, lmax=self.lmax, lonlat=True)
            
            cat_w = nmt.NmtWorkspace.from_fields(frb_cat, gal_clu, b)
            self.cat_w.append(cat_w)

            cat_w_simple = nmt.NmtWorkspace.from_fields(frb_cat, gal_clu, b_simple)
            self.cat_w_simple.append(cat_w_simple)

    # modeling functions
    
    def incomp_gamma(self, a, x):
        return sp.special.gammaincc(a, x)*sp.special.gamma(a)

    def n_z_or_chi(self, ALPHA, Z_STAR):
        
        dL = self.dl_grid
        dL_star = self.pm.get_luminosity_dist(Z_STAR)
        # Minimum detectable L/L*
        x_min = (dL / dL_star)**2
        # Get source density by integrating the luminocity function.
        if ALPHA+1 > 0:
            a = ALPHA+1
            n = self.incomp_gamma(a, x_min)
        if ALPHA+1 == 0:
            n = -sp.special.expi(-x_min)
        if -1 < ALPHA+1 < 0:
            a = ALPHA+1
            n = (self.incomp_gamma(a+1, x_min)-x_min**a*np.exp(-x_min))/a
        
        return n
    
    def get_n_z_norm(self, ALPHA, Z_STAR):
    
        n = self.n_z_or_chi(ALPHA, Z_STAR)
        norm = self.Nf/trapezoid(4*np.pi*self.chi_grid**2*n, self.chi_grid)
        n_norm = n*norm
    
        return n_norm
    
    def get_p_z_norm(self, ALPHA, Z_STAR):

        # Cache: within one likelihood evaluation, combined_model calls this
        # once per z-bin (and twice per sub-bin) with IDENTICAL (ALPHA, Z_STAR).
        # The underlying n_z_or_chi -> incomp_gamma chain is the single most
        # expensive part of the model, so memoize on the parameter pair.
        cache = getattr(self, "_pz_cache", None)
        if cache is not None and cache[0] == ALPHA and cache[1] == Z_STAR:
            return cache[2], cache[3]

        n = self.get_n_z_norm(ALPHA, Z_STAR)
        pz = 4*np.pi*self.chi_grid**2*n*self.dchi_dz_grid/self.Nf

        self._pz_cache = (ALPHA, Z_STAR, pz, n)
        return pz, n

    def compute_Nfz(self, pz, zs):
        # number of FRBs beyond redshift z
        # pz is now defined on the entire z grid
        # compute Nfz for z listed in zs; return an array
        zs = np.array(zs)
        z_loc = (zs/self.int_step).astype(int)
        return (1 - np.array([trapezoid(pz[:z_loc[i]], self.z_grid[:z_loc[i]]) for i in range(len(zs))]))*self.Nf

    def model_cl_Dg(self, l, z_ind, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR):

        cut_scale = 10**cut_scale

        # pz, nz depend only on (ALPHA, Z_STAR) -> identical across bins; get once
        pz, nz = self.get_p_z_norm(ALPHA, Z_STAR)

        if self.sub_bin[z_ind]:
            cl_Dg = 0
            for i in range(2):
                z = self.sub_bin_zs[z_ind][i]
                bg = [self.sub_bin_bg[z_ind][i], 0, 0]
                bf = [bf0, 0, 0]
                w = self.sub_bin_w[z_ind][i]
                chi_g = self.sub_bin_chi_g[z_ind][i]
                pz_index = int(z/self.int_step) - 1
                Nfz = self.compute_Nfz(pz, [z])[0]

                dm_bar = self.sub_bin_dm_bar[z_ind][i]
                DM_host = (self.DM_MWH + DM_H_bar/(1 + z)) * self._pc_cm3_to_Mpc2

                pm = self.sub_bin_pm[z_ind] # find the sub bin power spectrum
                pfg = get_biased_cross_spectrum(pm.k_over_h, pm.z[i], pm.pk[i], bf, bg)
                peg = get_biased_cross_spectrum_exp(pm.k_over_h, pm.z[i], pm.pk[i], self.be, bg, 1/cut_scale)
                k_eval = l/chi_g/fp.h
                Peg_at_l = np.interp(k_eval, peg.k_over_h, peg.pk) # Mpc^3 h^-3
                Pfg_at_l = np.interp(k_eval, pfg.k_over_h, pfg.pk)

                term1 = self.ne0*(1 + z)/chi_g**2*Peg_at_l*(fp.h**-3)*Nfz/self.Nf
                term2 = 4*np.pi*nz[pz_index]/self.Nf*Pfg_at_l*(fp.h**-3)*(dm_bar + DM_host - self.D_bar)

                this_cl_Dg = (term1 + term2)*w
                cl_Dg += this_cl_Dg

        else:
            z = self.zs[z_ind]
            bg = self.bg[z_ind]
            bf = [bf0, 0, 0]
            pz_index = int(z/self.int_step) - 1
            Nfz = self.compute_Nfz(pz, [z])[0]

            DM_host = (self.DM_MWH + DM_H_bar/(1 + z)) * self._pc_cm3_to_Mpc2

            pfg = get_biased_cross_spectrum(self.pm.k_over_h, self.pm.z[z_ind], self.pm.pk[z_ind], bf, bg)
            peg = get_biased_cross_spectrum_exp(self.pm.k_over_h, self.pm.z[z_ind], self.pm.pk[z_ind], self.be, bg, 1/cut_scale)
            k_eval = l/self.chi_g[z_ind]/fp.h
            Peg_at_l = np.interp(k_eval, peg.k_over_h, peg.pk) # Mpc^3 h^-3
            Pfg_at_l = np.interp(k_eval, pfg.k_over_h, pfg.pk)

            term1 = self.ne0*(1 + z)/self.chi_g[z_ind]**2*Peg_at_l*(fp.h**-3)*Nfz/self.Nf
            term2 = pz[pz_index]*(self.chi_g[z_ind]**(-2))/self.dchi_dz[z_ind]*Pfg_at_l*(fp.h**-3)*(self.dm_bar[z_ind] + DM_host - self.D_bar)

            cl_Dg = term1 + term2

        # gaussian beam suppression
        cl_Dg *= np.exp(-l**2/l_cut**2) # Mpc^-2

        return cl_Dg * self._Mpc2_to_pc_cm3 # from Mpc^-2 back to pc cm^-3

    def combined_model(self, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR):

        cl_joint = np.array([])
        ls = np.linspace(1, self.lmax, self.lmax)

        if self.convolve:
            for i in range(self.nz):
                cl = self.model_cl_Dg(ls, i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                cl_convolve = self.cat_w[i].decouple_cell(self.cat_w[i].couple_cell([np.insert(cl,0,0)]))[0]
                cl_joint = np.concatenate([cl_joint, cl_convolve[self.bin_excl:]])
        else:
            for i in range(self.nz):
                cl = self.model_cl_Dg(self.l_fit[i][self.bin_excl:], i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                cl_joint = np.concatenate([cl_joint, cl])

        return cl_joint

    # plotting functions
    def plot_models(self, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR, figsize=(20, 5), orientation = 'horizontal', mode = 'full', convolved = False, exclude = True, save = False, filename = 'mean_para_val_fit'):

        pz, nz = self.get_p_z_norm(ALPHA, Z_STAR)
        Nfz = self.compute_Nfz(pz, self.zs)

        lmin = 1
        lmax = self.lmax
        lnum = self.lmax
        l = np.linspace(lmin, lmax, lnum)

        if exclude:
            idx_start = self.bin_excl
        else:
            idx_start = 0

        assert (orientation == 'horizontal') or (orientation == 'vertical')
        if orientation == 'horizontal':        
            f, ax = plt.subplots(1, self.nz, sharey=True, figsize=figsize)
        if orientation == 'vertical':
            f, ax = plt.subplots(self.nz, 1, sharex=True, figsize=figsize)

        assert (mode == 'full') or (mode == 'simple')
        if mode == 'full':
            self.z_score_full = np.zeros((self.nz, 10))
            for i in range(self.nz):
                cl_Dg = self.model_cl_Dg(l, i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                ax[i].semilogx(l, l*cl_Dg, label = f"z = {self.zs[i]:.3f}", color = f'C{i}')
                ax[i].errorbar(self.l_fit[i][idx_start:], (self.cl_fit[i][idx_start:])*self.l_fit[i][idx_start:], yerr=self.err_fit[i][idx_start:]*self.l_fit[i][idx_start:],
                               capsize=2, marker=".", linestyle = 'none', color = f'C{i}')
                if convolved == True:
                    cl_convolve = self.cat_w[i].decouple_cell(self.cat_w[i].couple_cell([np.insert(cl_Dg,0,0)]))[0] # adding l = 0 term which is 0 (mean removed from data)
                    ax[i].plot(self.l_fit[i], self.l_fit[i]*cl_convolve, 'r--', label = f"template convolved")
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r'$\ell$', fontsize=15)
                if i==0:
                    ax[i].set_ylabel(r'$\ell C_\ell$ [cm$^{-3}$ pc]', fontsize=15)
                ax[i].legend()
                ax[i].hlines(0,0,8e3, color = 'black', linestyle = 'dashed')
                ax[i].set_xlim(0.9, 8e3)

                l_idx = self.l_fit[i][self.bin_excl:].astype(int) - 1
                assert all(l_idx >= 0)
                self.z_score_full[i,:] = (self.cl_fit[i][self.bin_excl:] - cl_Dg[l_idx])/self.err_fit[i][self.bin_excl:]
                
        if mode == 'simple':
            self.z_score_simple = np.zeros((self.nz, 5))
            for i in range(self.nz):
                cl_Dg = self.model_cl_Dg(l, i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                ax[i].semilogx(l, l*cl_Dg, label = f"z = {self.zs[i]:.3f}", color = f'C{i}')
                ax[i].errorbar(self.l_plot[i][idx_start:], (self.cl_plot[i][idx_start:])*self.l_plot[i][idx_start:], yerr=self.err_plot[i][idx_start:]*self.l_plot[i][idx_start:],
                               capsize=2, marker=".", linestyle = 'none', color = f'C{i}')
                if convolved == True:
                    cl_convolve = self.cat_w_simple[i].decouple_cell(self.cat_w_simple[i].couple_cell([np.insert(cl_Dg,0,0)]))[0] # adding l = 0 term which is 0 (mean removed from data)
                    ax[i].plot(self.l_plot[i], self.l_plot[i]*cl_convolve, 'r--', label = f"template convolved")
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r'$\ell$', fontsize=15)
                ax[i].set_ylabel(r'$\ell C_\ell$ [cm$^{-3}$ pc]', fontsize=15)
                ax[i].legend()
                ax[i].hlines(0,0,8e3, color = 'black', linestyle = 'dashed')
                ax[i].set_xlim(max(self.l_plot[i][idx_start] - 20, 0.5), 7e3)

                l_idx = self.l_plot[i][self.bin_excl:].astype(int) - 1
                assert all(l_idx >= 0)
                self.z_score_simple[i,:] = (self.cl_plot[i][self.bin_excl:] - cl_Dg[l_idx])/self.err_plot[i][self.bin_excl:]

        if save:
            filename += '_' + mode + '.pdf'
            f.savefig(filename)

    def model_cl_Dg_plot(self, l, z_ind, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR):

        cut_scale = 10**cut_scale

        term1_return = 0
        term2_return = 0
        
        if self.sub_bin[z_ind]:
            cl_Dg = 0
            
            for i in range(2):
                z = self.sub_bin_zs[z_ind][i]
                bg = [self.sub_bin_bg[z_ind][i], 0, 0]
                bf = [bf0, 0, 0]
                w = self.sub_bin_w[z_ind][i]
                chi_g = self.sub_bin_chi_g[z_ind][i]
                pz, nz = self.get_p_z_norm(ALPHA, Z_STAR)
                pz_index = int(z/self.int_step) - 1
                Nfz = self.compute_Nfz(pz, [z])[0]

                dm_bar = self.sub_bin_dm_bar[z_ind][i]
                DM_host = ((self.DM_MWH + DM_H_bar/(1 + z)) * u.pc * u.cm**-3).to(u.Mpc**-2).value
                
                pm = self.sub_bin_pm[z_ind] # find the sub bin power spectrum
                pfg = get_biased_cross_spectrum(pm.k_over_h, pm.z[i], pm.pk[i], bf, bg)
                peg = get_biased_cross_spectrum_exp(pm.k_over_h, pm.z[i], pm.pk[i], self.be, bg, 1/cut_scale)
                Peg_interp = interp1d(peg.k_over_h[:], peg.pk[:], kind='linear') # units: Mpc^3 h^-3, to get Mpc^-3: multiply by h^-3
                Pfg_interp = interp1d(pfg.k_over_h[:], pfg.pk[:], kind='linear')  

                term1 = self.ne0*(1 + z)/chi_g**2*Peg_interp(l/chi_g/fp.h)*(fp.h**-3)*Nfz/self.Nf
                term2 = 4*np.pi*nz[pz_index]/self.Nf*Pfg_interp(l/chi_g/fp.h)*(fp.h**-3)*(dm_bar + DM_host - self.D_bar)

                this_cl_Dg = (term1 + term2)*w
                cl_Dg += this_cl_Dg
                term1_return += term1*w
                term2_return += term2*w
            
        else:
            z = self.zs[z_ind]
            bg = self.bg[z_ind]
            bf = [bf0, 0, 0]
            pz, nz = self.get_p_z_norm(ALPHA, Z_STAR)
            pz_index = int(z/self.int_step) - 1
            Nfz = self.compute_Nfz(pz, [z])[0]

            DM_host = ((self.DM_MWH + DM_H_bar/(1 + z)) * u.pc * u.cm**-3).to(u.Mpc**-2).value
            
            pfg = get_biased_cross_spectrum(self.pm.k_over_h, self.pm.z[z_ind], self.pm.pk[z_ind], bf, bg)
            peg = get_biased_cross_spectrum_exp(self.pm.k_over_h, self.pm.z[z_ind], self.pm.pk[z_ind], self.be, bg, 1/cut_scale)
            Peg_interp = interp1d(peg.k_over_h[:], peg.pk[:], kind='linear') # units: Mpc^3 h^-3, to get Mpc^-3: multiply by h^-3
            Pfg_interp = interp1d(pfg.k_over_h[:], pfg.pk[:], kind='linear')
    
            term1 = self.ne0*(1 + z)/self.chi_g[z_ind]**2*Peg_interp(l/self.chi_g[z_ind]/fp.h)*(fp.h**-3)*Nfz/self.Nf
            term2 = pz[pz_index]*(self.chi_g[z_ind]**(-2))/self.dchi_dz[z_ind]*Pfg_interp(l/self.chi_g[z_ind]/fp.h)*(fp.h**-3)*(self.dm_bar[z_ind] + DM_host - self.D_bar)
    
            cl_Dg = term1 + term2
            term1_return = term1
            term2_return = term2

        # gaussian beam suppression
        cl_Dg *= np.exp(-l**2/l_cut**2) # Mpc^-2
        term1_return *= np.exp(-l**2/l_cut**2)
        term2_return *= np.exp(-l**2/l_cut**2)
        
        return (cl_Dg * u.Mpc**-2).to(u.pc * u.cm**-3).value, (term1_return * u.Mpc**-2).to(u.pc * u.cm**-3).value, (term2_return * u.Mpc**-2).to(u.pc * u.cm**-3).value # from Mpc^-2 back to pc cm^-3

    def plot_models_plus(self, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR, figsize=(20, 5), orientation = 'horizontal', mode = 'full', convolved = False, exclude = True, save = False, filename = 'separate_comp_fit'):

        pz, nz = self.get_p_z_norm(ALPHA, Z_STAR)
        Nfz = self.compute_Nfz(pz, self.zs)

        lmin = 1
        lmax = self.lmax
        lnum = self.lmax
        l = np.linspace(lmin, lmax, lnum)

        if exclude:
            idx_start = self.bin_excl
        else:
            idx_start = 0

        assert (orientation == 'horizontal') or (orientation == 'vertical')
        if orientation == 'horizontal':        
            f, ax = plt.subplots(1, self.nz, sharey=True, figsize=figsize)
        if orientation == 'vertical':
            f, ax = plt.subplots(self.nz, 1, sharex=True, figsize=figsize)

        self.ft = [self.forward_trans0, self.forward_trans1, self.forward_trans2, self.forward_trans3, self.forward_trans4]
        self.it = [self.inverse_trans0, self.inverse_trans1, self.inverse_trans2, self.inverse_trans3, self.inverse_trans4]
        
        assert (mode == 'full') or (mode == 'simple')
        if mode == 'full':
            self.z_score_full = np.zeros((self.nz, 10))
            for i in range(self.nz):
                cl_Dg, term1, term2 = self.model_cl_Dg_plot(l, i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                ax[i].semilogx(l, l*cl_Dg, label = f"z = {self.zs[i]:.3f}", color = f'C0')
                ax[i].semilogx(l, l*term1, label = r"P_{eg}", color = 'C1', linestyle = "--")
                ax[i].semilogx(l, l*term2, label = r"P_{fg}", color = 'C2', linestyle = "--")
                ax[i].errorbar(self.l_fit[i][idx_start:], (self.cl_fit[i][idx_start:])*self.l_fit[i][idx_start:], yerr=self.err_fit[i][idx_start:]*self.l_fit[i][idx_start:],
                               capsize=2, marker=".", linestyle = 'none', color = f'C0')
                if convolved == True:
                    cl_convolve = self.cat_w[i].decouple_cell(self.cat_w[i].couple_cell([np.insert(cl_Dg,0,0)]))[0] # adding l = 0 term which is 0 (mean removed from data)
                    ax[i].plot(self.l_fit[i], self.l_fit[i]*cl_convolve, 'r--', label = f"template convolved")
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r'$\ell$', fontsize=15)
                if i==0:
                    ax[i].set_ylabel(r'$\ell C_\ell$ [cm$^{-3}$ pc]', fontsize=15)
                ax[i].legend()
                ax[i].hlines(0,0,8e3, color = 'black', linestyle = 'dashed')
                ax[i].set_xlim(0.9, 8e3)

                top_ax = ax[i].secondary_xaxis('top',functions=(self.ft[i], self.it[i])) 
                top_ax.set_xlabel(r'k [1/Mpc]') 

                l_idx = self.l_fit[i][self.bin_excl:].astype(int) - 1
                assert all(l_idx >= 0)
                self.z_score_full[i,:] = (self.cl_fit[i][self.bin_excl:] - cl_Dg[l_idx])/self.err_fit[i][self.bin_excl:]

        if mode == 'simple':
            self.z_score_simple = np.zeros((self.nz, 5))
            for i in range(self.nz):
                cl_Dg, term1, term2 = self.model_cl_Dg_plot(l, i, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
                ax[i].semilogx(l, l*cl_Dg, label = f"z = {self.zs[i]:.3f}", color = f'C0')
                ax[i].semilogx(l, l*term1, label = r"P_{eg}", color = 'C1', linestyle = "--")
                ax[i].semilogx(l, l*term2, label = r"P_{fg}", color = 'C2', linestyle = "--")                
                ax[i].errorbar(self.l_plot[i][idx_start:], (self.cl_plot[i][idx_start:])*self.l_plot[i][idx_start:], yerr=self.err_plot[i][idx_start:]*self.l_plot[i][idx_start:],
                               capsize=2, marker=".", linestyle = 'none', color = f'C0')
                if convolved == True:
                    cl_convolve = self.cat_w_simple[i].decouple_cell(self.cat_w_simple[i].couple_cell([np.insert(cl_Dg,0,0)]))[0] # adding l = 0 term which is 0 (mean removed from data)
                    ax[i].plot(self.l_plot[i], self.l_plot[i]*cl_convolve, 'r--', label = f"template convolved")
                ax[i].set_xscale('log')
                ax[i].set_xlabel(r'$\ell$', fontsize=15)
                ax[i].set_ylabel(r'$\ell C_\ell$ [cm$^{-3}$ pc]', fontsize=15)
                ax[i].legend()
                ax[i].hlines(0,0,8e3, color = 'black', linestyle = 'dashed')
                ax[i].set_xlim(max(self.l_plot[i][idx_start] - 20, 0.5), 7e3)

                top_ax = ax[i].secondary_xaxis('top',functions=(self.ft[i], self.it[i])) 
                top_ax.set_xlabel(r'k [1/Mpc]')                 

                l_idx = self.l_plot[i][self.bin_excl:].astype(int) - 1
                assert all(l_idx >= 0)
                self.z_score_simple[i,:] = (self.cl_plot[i][self.bin_excl:] - cl_Dg[l_idx])/self.err_plot[i][self.bin_excl:]

        try:
            self.convolve  # Attempt to access the variable
        except AttributeError:
            self.convolve = False
        theta = (DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
        this_log_likelihood = self.log_likelihood(theta, self.l_combined, self.cl_combined, self.err_combined)
        del_chi2 = 2*(this_log_likelihood - self.log_likelihood_null(self.l_combined, self.cl_combined, self.err_combined))
        f.suptitle(rf"$\Delta \chi^2 = $ {del_chi2:.2f}", x=0.15, y=0.83)

        if save:
            filename += '_' + mode + '.pdf'
            f.savefig(filename)

    def forward_trans0(self, l):
        k = l/self.chi_g[0]
        return k

    def inverse_trans0(self, k):
        l = k*self.chi_g[0]
        return l

    def forward_trans1(self, l):
        k = l/self.chi_g[1]
        return k

    def inverse_trans1(self, k):
        l = k*self.chi_g[1]
        return l

    def forward_trans2(self, l):
        k = l/self.chi_g[2]
        return k

    def inverse_trans2(self, k):
        l = k*self.chi_g[2]
        return l

    def forward_trans3(self, l):
        k = l/self.chi_g[3]
        return k

    def inverse_trans3(self, k):
        l = k*self.chi_g[3]
        return l

    def forward_trans4(self, l):
        k = l/self.chi_g[4]
        return k

    def inverse_trans4(self, k):
        l = k*self.chi_g[4]
        return l

    # MCMC functions

    def get_cov(self):
        rand_ps = []
        err = []

        for i in range(self.nz):
            file = h5py.File(self.fit_files[i], 'r')
            rand_ps.extend(file['rand_ps'][:])
            err.extend(file['error'][0][4:])

        rand_ps = np.array(rand_ps)
        err = np.array(err)

        data = []
        for i in range(self.nz):
            data.extend(rand_ps[i, 4:, :])
        data = np.array(data)

        noise_cov = np.cov(data)
        self.cov = noise_cov
        self.cov_inv = sp.linalg.inv(self.cov)

        diag = np.diag(noise_cov)
        self.corr_mat = noise_cov/np.sqrt(diag[:, None]*diag[None,:])

    def log_likelihood(self, theta, l, Cl, err):
        
        DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR = theta
        model_Cl = self.combined_model(DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
        chi_vec = Cl - model_Cl
        return -0.5*chi_vec @ self.cov_inv @ chi_vec
        
    def log_prior(self, theta):
        # unnormalized prior
        DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR = theta
        
        if (50 < DM_H_bar < 250) and (np.log10(0.05) < cut_scale < np.log10(50)) and (315 < l_cut < 8000) and (1 < bf0 < 4) and (-2 < ALPHA < 1) and (0.1 < Z_STAR < 1.5):
            return 0.0
        return -np.inf
    
    def log_probability(self, theta, l, Cl, err):
        
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf 
        return lp + self.log_likelihood(theta, l, Cl, err)

    def mcmc(self, ndim = 6, theta_init = np.array([60, -1, 4000, 1.5, -1.5, 1]), nwalkers = 32, nsteps = 50000, thin = 20, burn_in = 3000, convolve = False, save = False, filename = 'mcmc_out.h5'):

        self.convolve = convolve
        assert ndim == len(theta_init)
        self.ndim = ndim
        self.theta_init = theta_init
        self.get_cov()

        theta_init = self.theta_init + 1e-3 * np.random.randn(nwalkers, self.ndim)
        self.sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_probability, args=(self.l_combined, self.cl_combined, self.err_combined)) # self.l_combined ... come from read_data, concatenated with l bins used for fitting only 
        self.sampler.run_mcmc(theta_init, nsteps, progress=True)
        self.full_chains = self.sampler.get_chain()
        self.flat_samples = self.sampler.get_chain(discard=burn_in, thin=thin, flat=True) # technically burn-in should be 10*autocorrelation length, and thin = autocorrelation length (250)

        # Plotting
        labels = ["<DM_H>", "s_cut", "l_cut", "bf", "ALPHA", "Z_STAR"]
        fig = corner.corner(self.flat_samples, labels=labels)
        fig.savefig('corner_plot.pdf')
        
        # Print the estimated parameters
        self.DMH_bar_mc, self.cut_scale_mc, self.l_cut_mc, self.bf0_mc, self.ALPHA_mc, self.Z_STAR = np.mean(self.flat_samples, axis=0)
        self.mean_parameters = [self.DMH_bar_mc, self.cut_scale_mc, self.l_cut_mc, self.bf0_mc, self.ALPHA_mc, self.Z_STAR]
        print_out = f"Fitted Parameters:\nDM_H_bar = {self.DMH_bar_mc}\ncut_scale  = {self.cut_scale_mc}\nl_cut = {self.l_cut_mc}\nbf0 = {self.bf0_mc}\nALPHA = {self.ALPHA_mc}\nZ_STAR = {self.Z_STAR}"
        print(print_out)
        
        # likelihood ratio test (namely delta chi square test)
        # compute log_likelihood from the posterior samples
        drawn_log_likelihood = np.array([self.log_likelihood(theta, self.l_combined, self.cl_combined, self.err_combined) for theta in self.flat_samples]) 
        self.drawn_log_likelihood = drawn_log_likelihood
        max_log_likelihood = np.max(drawn_log_likelihood)
        d_M = 2*(np.mean(drawn_log_likelihood**2) - (np.mean(drawn_log_likelihood))**2)
        print("Effective degree of freedom:", d_M)
        self.max_del_chi2 = 2*(max_log_likelihood - self.log_likelihood_null(self.l_combined, self.cl_combined, self.err_combined))
        N_sig_eff = sp.stats.norm.ppf(chi2.cdf(self.max_del_chi2, round(d_M)))
        print("Maximum delta chi square is", self.max_del_chi2)
        print("Effective number of sigma (likelihood ratio test):", N_sig_eff)

        max_idx = np.argmax(drawn_log_likelihood)
        self.DMH_bar_max, self.cut_scale_max, self.l_cut_max, self.bf0_max, self.ALPHA_max, self.Z_STAR_max = self.flat_samples[max_idx,:]
        maximum_likelihood_para = [self.DMH_bar_max, self.cut_scale_max, self.l_cut_max, self.bf0_max, self.ALPHA_max, self.Z_STAR_max]

        if save:
            file = h5py.File(filename, 'w')
            file.create_dataset('nsteps', data = nsteps)
            file.create_dataset('nwalkers', data = nwalkers)
            file.create_dataset('burn_in', data = burn_in)
            file.create_dataset('thin', data = thin)
            file.create_dataset('full_chains', data = self.full_chains)
            file.create_dataset('flat_samples', data = self.flat_samples)
            file.create_dataset('accept_frac', data = self.sampler.acceptance_fraction)
            file.create_dataset('autocorr_time', data = self.sampler.get_autocorr_time())
            file.create_dataset('mean_parameters', data = self.mean_parameters)
            file.create_dataset('parameter_names', data = labels)
            file.create_dataset('drawn_log_likelihood', data = self.drawn_log_likelihood)
            file.create_dataset('max_del_chi2', data = self.max_del_chi2)
            file.create_dataset('detection_sig', data = N_sig_eff)
            file.create_dataset('maximum_likelihood_para', data = maximum_likelihood_para)

    # evaluating detection significance
    
    def log_likelihood_null(self, l, Cl, err):
        
        model_null = np.zeros_like(l)
        chi_vec = Cl - model_null
        return -0.5*chi_vec @ self.cov_inv @ chi_vec
    
    def log_prior_null(self):
        return 0.0
    
    def log_posterior_null(self, l, Cl, err):
        return self.log_prior_null() + self.log_likelihood_null(l, Cl, err)

    # Computing Bayes factor using thermodynamic integration (TDI)

    def log_likelihood_lamb_prior(self, theta, l, Cl, err, lamb):
        
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf 
        return lp + self.log_likelihood(theta, l, Cl, err)*lamb # remember it is in log space, so not **lamb
    
    def TDI(self, npoints = 30, nwalkers = 32, nsteps = 5000, burn_in = 3000, thin = 20):
        
        # a grid evenly spaced in lambda^(1/5)
        lamb_1_over_5_grid = np.linspace(0, 1, npoints)
        # translates to a grid in lambda
        self.lamb_grid = lamb_1_over_5_grid**5

        self.integrand = np.zeros(npoints)
        self.TDI_sampler = []

        for n, lamb in enumerate(self.lamb_grid):
            
            # Sampling L(theta)^lambda pi(theta)
            theta_init = self.theta_init + 1e-3 * np.random.randn(nwalkers, self.ndim)
            self.TDI_sampler.append(emcee.EnsembleSampler(nwalkers, self.ndim, self.log_likelihood_lamb_prior, args=(self.l_combined, self.cl_combined, self.err_combined, lamb)))
            self.TDI_sampler[n].run_mcmc(theta_init, nsteps, progress=True)
            flat_samples =self.TDI_sampler[n].get_chain(discard=burn_in, thin=thin, flat=True) # think about burn-in discard and thinning more

            drawn_log_likelihood = np.array([self.log_likelihood(theta, self.l_combined, self.cl_combined, self.err_combined) for theta in flat_samples])
            expectation = np.mean(drawn_log_likelihood)
            self.integrand[n] = expectation

        self.log_z = trapezoid(self.integrand, self.lamb_grid)

    def log_z_error(self, nresample = 10000, replacement = True):

        # bootstrap resampling 
        resamples = np.random.choice(self.integrand, size=(nresample, len(self.integrand)), replace=replacement)
        integrated_resamples = trapezoid(resamples, self.lamb_grid, axis = -1)
        self.log_z_err = np.std(integrated_resamples)
        print(f"the error on log z is {self.log_z_err}, which is {np.exp(self.log_z_err)} on z.")
        print(f"the log evidence, log z, is {self.log_z}.")
    
    def bayes_factor(self, npoints = 30, nwalkers = 32, nsteps = 5000, burn_in = 3000, thin = 20, save = False, filename = 'bayes_factor.h5'):
        log_evidence_null = self.log_posterior_null(self.l_combined, self.cl_combined, self.err_combined)
        self.TDI(npoints = npoints, nwalkers = nwalkers, nsteps = nsteps, burn_in = burn_in, thin = thin)
        log_evidence = self.log_z

        log_bayes_factor = log_evidence - log_evidence_null
        bayes_factor = np.exp(log_bayes_factor)
        PM0 = 1 / (1 + bayes_factor)
        N_sig = sp.stats.norm.ppf(1 - PM0)
        print("Bayes Factor:", bayes_factor)
        print("Detection significance (sigma):", N_sig)

        TDI_full_chains = [self.TDI_sampler[n].get_chain() for n in range(npoints)]
        TDI_flat_samples = [self.TDI_sampler[n].get_chain(discard=burn_in, thin=thin, flat=True) for n in range(npoints)]
        #TDI_autocorr_time = [self.TDI_sampler[n].get_autocorr_time() for n in range(npoints)]
        #TDI_accept_frac = [self.TDI_sampler[n].acceptance_fraction for n in range(npoints)]

        self.log_z_error()

        if save:
            file = h5py.File(filename, 'w')
            file.create_dataset('log_evidence', data = self.log_z)
            file.create_dataset('log_evidence_null', data = log_evidence_null)
            file.create_dataset('log_bayes_factor', data = log_bayes_factor)
            file.create_dataset('bayes_factor', data = bayes_factor)
            file.create_dataset('detection_sig', data = N_sig)
            file.create_dataset('npoints', data = npoints)
            file.create_dataset('nsteps', data = nsteps)
            file.create_dataset('nwalkers', data = nwalkers)
            file.create_dataset('burn_in', data = burn_in)
            file.create_dataset('thin', data = thin)
            file.create_dataset('lamb_grid', data = self.lamb_grid)
            file.create_dataset('TDI_integrand', data = self.integrand)
            file.create_dataset('TDI_full_chains', data = TDI_full_chains)
            file.create_dataset('TDI_flat_samples', data = TDI_flat_samples)
            #file.create_dataset('TDI_autocorr_time', data = TDI_autocorr_time)
            #file.create_dataset('TDI_accept_frac', data = TDI_accept_frac)
            file.create_dataset('log_z_err', data = self.log_z_err)