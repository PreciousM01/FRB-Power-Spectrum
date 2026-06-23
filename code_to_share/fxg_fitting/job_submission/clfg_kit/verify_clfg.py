"""Compare ORIGINAL (mcmcfit_clfg_orig) vs OPTIMIZED (mcmcfit_clfg) model output.
Both must produce the same combined_model() result on the same params/data.
Run from inside clfg_kit:  python verify_clfg.py
"""
import numpy as np, importlib
import matplotlib; matplotlib.use("Agg")

PS_FILE = "./dummy_ps.h5"
d_bin_edge = np.array([38.17842193, 209.94991961, 291.8247406, 361.18080268,
                       444.52819007, 526.88686278, 620.77482314, 732.37124655,
                       886.70586238, 1155.88351684, 3916.20665172])
Nd = [288, 287, 287, 287, 287, 288, 287, 287, 287, 288]

# 7 params: mu_h, sig_h, cut_scale, l_cut, bf0, ALPHA, Z_STAR  (within prior bounds)
theta = (1.48, -0.058, 1.0, 1500.0, 2.0, -1.0, 0.5)

def model_out(modname):
    m = importlib.import_module(modname)
    g = m.cl_fg_fit(d_bin_edge, Nd)
    g.read_data(PS_FILE)
    return g.combined_model(*theta)

out0 = model_out("mcmcfit_clfg_orig")   # original
out1 = model_out("mcmcfit_clfg")        # optimized

abs_diff = np.max(np.abs(out0 - out1))
# relative diff guarded against divide-by-zero
denom = np.where(out0 == 0, 1.0, out0)
rel_diff = np.max(np.abs((out0 - out1)/denom))

print("shape:", out0.shape)
print("max abs diff:", abs_diff)
print("max rel diff:", rel_diff)
print("identical:", np.allclose(out0, out1, rtol=1e-10, atol=1e-30))
