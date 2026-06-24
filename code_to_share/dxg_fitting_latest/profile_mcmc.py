"""
profile_mcmc.py  --  Profile the MCMC sampling loop.

WHAT THIS DOES:
  1. Loads the (dummy) FRB catalog to get `dm_select` -> constructs cl_Dg_fit
  2. Loads the (dummy) power-spectrum files via read_data
  3. Runs a SHORT MCMC (150 steps) wrapped in cProfile
  4. Prints the 30 slowest functions, sorted by tottime (time IN each function)
  5. Saves raw stats to mcmc_profile.prof so you can re-slice later

HOW THE VARIABLES CONNECT (this is the part that was unclear):
  - dm_select   : 1D array of FRB dispersion measures. In the real code this
                  comes from your FRB catalog .h5. cl_Dg_fit needs it to build
                  the FRB number density. Here it's read from the dummy catalog.
  - fit_files   : list of 5 .h5 files (one per redshift bin) holding the
                  measured cross-power-spectrum used for FITTING. read_data
                  loads bin_centers/cl/error/bin_edges/lmax from each, and
                  get_cov() (called inside mcmc) reads rand_ps for covariance.
  - plot_files  : list of 5 .h5 files (9-bin version) used only for plotting.
                  read_data still requires them, so we pass dummy ones.
  - theta_init  : starting point for the 6 fit parameters
                  [<DM_H>, s_cut, l_cut, bf, ALPHA, Z_STAR].

WHY 150 STEPS: profiling cares about PER-CALL cost. 150 steps x 32 walkers
  is ~4800 likelihood evaluations -- plenty to see what's slow -- and runs in
  ~1 minute instead of hours. The slow functions are the same at 150 or 50000.
"""
import numpy as np
import h5py
import cProfile
import pstats
import matplotlib
matplotlib.use("Agg")   # headless: corner.corner won't try to open a window

import mcmcfit as mf

# ---------------------------------------------------------------------------
# 1. Point these at your data. (Here: the dummy data in ./dummy_data)
#    When you have REAL data, change DATA_DIR and the file names to match.
# ---------------------------------------------------------------------------
DATA_DIR = "./dummy_data"
zs = [0.075, 0.15, 0.25, 0.35, 0.45]

with h5py.File(f"{DATA_DIR}/frb_catalog.h5", "r") as fh:
    mean_DM   = fh["dm_ave"][()]
    dm_select = fh["dm_od"][:] + mean_DM

fit_files  = [f"{DATA_DIR}/fit_z{z:.3f}.h5"  for z in zs]
plot_files = [f"{DATA_DIR}/plot_z{z:.3f}.h5" for z in zs]

# ---------------------------------------------------------------------------
# 2. Build the fitter object and load the data (same as curve_fit.py does)
# ---------------------------------------------------------------------------
f = mf.cl_Dg_fit(dm_select)
f.read_data(fit_files, plot_files)

# ---------------------------------------------------------------------------
# 3. Profile a SHORT real sampling run
# ---------------------------------------------------------------------------
with cProfile.Profile() as pr:
    f.mcmc(theta_init=np.array([60, -1, 4000, 1.5, -1.5, 1]),
           nwalkers=32, nsteps=150, thin=20, burn_in=100, save=False)

# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------
stats = pstats.Stats(pr)
stats.dump_stats("mcmc_profile.prof")    # raw, reload anytime with pstats
stats.sort_stats("tottime")              # sort by time spent INSIDE each function
print("\n================ TOP 30 BY tottime ================")
stats.print_stats(30)