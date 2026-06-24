"""
plot_dxg_models.py -- Use the built-in plot_models() on the REAL dxg_measurement data.

plot_models() makes the 5-panel (one per z bin) model-vs-data figure itself.
We call it with save=False and grab the figure to save as PNG.
Also plots pf(chi).

Edit DATA_DIR, file order, FRB_CATALOG, and the 6 params as needed.
"""
import numpy as np
import h5py
import glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mcmcfit as mf

# --- 1. real measurement files (redshift order, low z first) ---
DATA_DIR = "../dxg_measurement"                 # <-- EDIT
FILES = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
assert len(FILES) == 5, f"expected 5 files, found {len(FILES)}"
print("Using:", [os.path.basename(f) for f in FILES])
fit_files = FILES
plot_files = FILES                              # <-- EDIT if separate plot files

# --- 2. dm_select ---
FRB_CATALOG = None                              # <-- EDIT if you have cat_II_..._inclusion.h5
if FRB_CATALOG:
    with h5py.File(FRB_CATALOG, "r") as fh:
        dm_select = fh["dm_od"][:] + fh["dm_ave"][()]
else:
    dm_select = np.random.default_rng(0).normal(500, 200, 500)
    print("WARNING: stand-in dm_select (no real FRB catalog).")

# --- 3. build + load ---
g = mf.cl_Dg_fit(dm_select)
g.read_data(fit_files, plot_files)

# initial params: DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR
params = (60.0, -1.0, 4000.0, 1.5, -1.5, 1.0)
ALPHA, Z_STAR = params[4], params[5]

# --- pf(chi) ---
pf_z, n = g.get_p_z_norm(ALPHA, Z_STAR)
pf_chi = 4*np.pi*g.chi_grid**2*n/g.Nf
plt.figure()
plt.plot(g.chi_grid, pf_chi)
plt.xlabel(r'$\chi$ [Mpc]'); plt.ylabel(r'$p_f(\chi)$ [Mpc$^{-1}$]')
plt.title(f'FRB radial distribution (ALPHA={ALPHA}, Z_STAR={Z_STAR})')
plt.savefig("pf_dxg_real.png", dpi=150, bbox_inches="tight")
plt.close()

# --- model vs data via the built-in plot_models (save=False, we grab the fig) ---
g.plot_models(*params, mode='full', convolved=False, save=False)
plt.gcf().savefig("cl_dxg_models.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved: pf_dxg_real.png and cl_dxg_models.png")