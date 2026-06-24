"""
plot_code_outputs_dxg.py

Two plots:
  1. pf(chi): FRB radial distribution, from get_p_z_norm(ALPHA, Z_STAR)
  2. Cl: model power spectrum from model_cl_Dg_plot(...) for one z bin, split into
        total / background / same-halo, at the initial parameters (no fit run).

cl_Dg has 5 redshift bins (no DM bins) and 6 params:
  DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR
Uses dummy data for the FRB catalog + power-spectrum files.
"""
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mcmcfit as mf

DATA_DIR = "./dummy_data"
zs = [0.075, 0.15, 0.25, 0.35, 0.45]

# initial params, same order as mcmc theta_init: DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR
DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR = 60.0, -1.0, 4000.0, 1.5, -1.5, 1.0

# ---- setup (same as profile_mcmc.py) ----
with h5py.File(f"{DATA_DIR}/frb_catalog.h5", "r") as fh:
    dm_select = fh["dm_od"][:] + fh["dm_ave"][()]
g = mf.cl_Dg_fit(dm_select)
fit_files  = [f"{DATA_DIR}/fit_z{z:.3f}.h5"  for z in zs]
plot_files = [f"{DATA_DIR}/plot_z{z:.3f}.h5" for z in zs]
g.read_data(fit_files, plot_files)

# ---- Plot 1: pf(chi) -- convert from pf(z) so it matches cl_fg's get_p_chi_norm ----
pf_z, n = g.get_p_z_norm(ALPHA, Z_STAR)
pf_chi = 4*np.pi*g.chi_grid**2*n/g.Nf      # density in chi (drops the dchi/dz Jacobian)
plt.figure()
plt.plot(g.chi_grid, pf_chi)
plt.xlabel(r'$\chi$ [Mpc]')
plt.ylabel(r'$p_f(\chi)$ [Mpc$^{-1}$]')
plt.title(f'FRB radial distribution pf(chi) (cl_Dg, ALPHA={ALPHA}, Z_STAR={Z_STAR})')
plt.savefig("pf_dxg.png", dpi=150, bbox_inches="tight")

# ---- Plot 2: Cl for one z bin, split into total/background/same-halo ----
z_ind = 2                                  # z = 0.25 (single-redshift bin, not a sub-bin)
l = np.linspace(1, g.lmax, g.lmax)
cl_Dg, term1, term2 = g.model_cl_Dg_plot(l, z_ind, DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
plt.figure()
plt.semilogx(l, l*cl_Dg, label='total')
plt.semilogx(l, l*term1, '--', label='background')
plt.semilogx(l, l*term2, '-.', label='same halo')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell C_\ell$')
plt.title(f'Model $C_\\ell$ (cl_Dg, z bin {z_ind}, z={g.zs[z_ind]})')
plt.legend()
plt.savefig("cl_dxg.png", dpi=150, bbox_inches="tight")

print("Saved: pf_dxg.png and cl_dxg.png")