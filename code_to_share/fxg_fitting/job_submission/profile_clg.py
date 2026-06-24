import numpy as np, cProfile, pstats
import matplotlib; matplotlib.use("Agg")
import mcmcfit_clfg as mf

PS_FILE = "./mcmc_out_updated_signal_lognorm.h5" 

d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268,
                       444.52819007, 526.88686278, 620.77482314, 732.37124655,
                       886.70586238, 1155.88351684, 3916.20665172])
Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

g = mf.cl_fg_fit(d_bin_edge, Nd)
g.read_data(PS_FILE)

with cProfile.Profile() as pr:
    g.mcmc(save=False, thin=10, burn_in=100, nsteps=150)   # SHORT run

stats = pstats.Stats(pr)
stats.dump_stats("clfg_profile.prof")
stats.sort_stats("tottime")
print("\n================ TOP 30 BY tottime ================")
stats.print_stats(30)