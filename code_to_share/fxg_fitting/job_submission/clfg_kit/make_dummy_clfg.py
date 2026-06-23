"""Generate a dummy ps_file for profiling cl_fg_fit (shapes matched to read_data)."""
import numpy as np, h5py
np.random.seed(0)

n_d_bins = 10
nz = 5
n_lbins = 14
LMAX = 2771

# l-bin centers spread over the multipole range, plus matching edges
edges = np.linspace(1, LMAX, n_lbins + 1)
bin_centers = 0.5*(edges[:-1] + edges[1:])

# cl/error on disk: shape (nz, n_d_bins, n_lbins); read_data does swapaxes(0,1)->(n_d_bins,nz,n_lbins)
cl    = np.random.normal(0, 1e-3, (nz, n_d_bins, n_lbins))
error = np.abs(np.random.normal(1e-3, 1e-4, (nz, n_d_bins, n_lbins)))

with h5py.File("dummy_ps.h5", "w") as f:
    f.create_dataset("bin_edges",   data=edges)
    f.create_dataset("bin_centers", data=bin_centers)
    f.create_dataset("lmax",        data=np.int64(LMAX))
    f.create_dataset("n_dm_b",      data=np.int64(n_d_bins))
    f.create_dataset("n_z_b",       data=np.int64(nz))
    f.create_dataset("cl",          data=cl)
    f.create_dataset("error",       data=error)
print("wrote dummy_ps.h5")
