"""
run_and_plot.py -- Run the cl_fg MCMC on DUMMY data locally and make plots.

Mirrors the author's run_fit.py workflow but small enough to run on a laptop:
  1. build cl_fg_fit, read dummy power spectrum
  2. run a SHORT mcmc (makes the corner plot automatically)
  3. make the model-vs-data overlay plot using the fitted median params

NOTE: dummy data has no real signal, so the plots will look like noise.
The point is to confirm the plotting machinery runs and produces PDFs --
not that the science looks right.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")              # save to file instead of opening a window
import matplotlib.pyplot as plt
import mcmcfit_clfg as mf

PS_FILE = "./dummy_ps.h5"
d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268,
                       444.52819007, 526.88686278, 620.77482314, 732.37124655,
                       886.70586238, 1155.88351684, 3916.20665172])
Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

# 1. setup
g = mf.cl_fg_fit(d_bin_edge, Nd)
g.read_data(PS_FILE)

# 2. run a short chain (>=200 steps so corner has enough samples)
#    this saves corner_plot_updated_signal_lognorm.pdf automatically
g.mcmc(nsteps=400, burn_in=100, thin=2, save=False)

# 3. model-vs-data overlay using the fitted medians
#    med_parameters = [mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR]
g.plot_full_model_data(*g.med_parameters, figsize=(25, 25), exclude=True)
plt.savefig("model_vs_data.png", bbox_inches="tight")
print("\nSaved: corner_plot_updated_signal_lognorm.png  and  model_vs_data.pdf")
