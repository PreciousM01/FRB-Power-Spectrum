"""
profile_clfg.py -- Profile the cl_fg_fit MCMC sampling loop.

Same pattern as profile_mcmc.py but for the FRB-galaxy (cl_fg) fit:
construct cl_fg_fit(d_bin_edge, Nd), read_data(ps_file), then profile a
SHORT mcmc() run. Sorted by tottime; raw stats saved to clfg_profile.prof.
"""
import numpy as np, cProfile, pstats
import matplotlib; matplotlib.use("Agg")   # headless; corner.corner won't open a window
import mcmcfit_clfg as mf

PS_FILE = "./dummy_ps.h5"   # <-- change to your real ps_file path when you have data

# Same DM-bin setup as curve_fit_clfg.py driver
d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268,
                       444.52819007, 526.88686278, 620.77482314, 732.37124655,
                       886.70586238, 1155.88351684, 3916.20665172])
Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

g = mf.cl_fg_fit(d_bin_edge, Nd)
g.read_data(PS_FILE)

with cProfile.Profile() as pr:
    g.mcmc(save=False, thin=10, burn_in=100, nsteps=150)   # SHORT run for profiling

stats = pstats.Stats(pr)
stats.dump_stats("clfg_profile.prof")
stats.sort_stats("tottime")
print("\n================ TOP 30 BY tottime ================")
stats.print_stats(30)
