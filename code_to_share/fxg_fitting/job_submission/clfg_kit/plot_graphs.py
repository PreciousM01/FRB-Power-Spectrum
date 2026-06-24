"""
Two plots:
  1. pf(chi): the FRB radial distribution, straight from get_p_chi_norm(ALPHA, Z_STAR)
  2. Cl: the model power spectrum from model_cl_fg(...) for one (d,z) bin,
        evaluated at the initial parameters (no fit run).

Uses dummy data for the power-spectrum file. Parameters are the defaults from
mcmc()'s theta_init: [mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR].
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mcmcfit_clfg as mf

PS_FILE = "./dummy_ps.h5"
d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268,
                       444.52819007, 526.88686278, 620.77482314, 732.37124655,
                       886.70586238, 1155.88351684, 3916.20665172])
Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

# initial params, same order/defaults as mcmc(): mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR
mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR = 1.48, -0.058, 1.0, 1500.0, 2.0, -1.0, 0.5

g = mf.cl_fg_fit(d_bin_edge, Nd)
g.read_data(PS_FILE)

# ---- Plot 1: pf(chi), exactly as get_p_chi_norm returns ----
pf_chi, n = g.get_p_chi_norm(ALPHA, Z_STAR)   # arrays along g.chi_grid
plt.figure()
plt.plot(g.chi_grid, pf_chi)
plt.xlabel(r'$\chi$ [Mpc]')
plt.ylabel(r'$p_f(\chi)$ [Mpc$^{-1}$]')
plt.title('FRB radial distribution (optimized code output)')
plt.savefig("pf_chi.png", bbox_inches="tight")

# ---- Plot 2: Cl, exactly as model_cl_fg returns, for one (d,z) bin ----
d_ind, z_ind = 0, 0
l = np.linspace(1, g.lmax, g.lmax)
cl_fg, term1, term2 = g.model_cl_fg(l, d_ind, z_ind, mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR)
plt.figure()
plt.semilogx(l, l*cl_fg, label='total')
plt.semilogx(l, l*term1, '--', label='background')
plt.semilogx(l, l*term2, '-.', label='same halo')
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell C_\ell$')
plt.title(f'Model $C_\\ell$ (DM bin {d_ind}, z bin {z_ind}) (optimized code output)')
plt.legend()
plt.savefig("cl_model.png", bbox_inches="tight")

print("Saved: pf_chi.pdf and cl_model.pdf")
print("pf_chi: array len", len(pf_chi), "over chi_grid len", len(g.chi_grid))
print("cl_fg:  array len", len(cl_fg), "over l len", len(l))
