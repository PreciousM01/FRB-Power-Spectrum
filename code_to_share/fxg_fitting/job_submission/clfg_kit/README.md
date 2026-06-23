# Profiling kit for mcmcfit_clfg.py (cl_fg_fit)

Profile the FRB-galaxy MCMC WITHOUT real data, using a dummy input file.

## Files
    mcmcfit_clfg.py        <- OPTIMIZED code (2 caches applied; verified identical)
    mcmcfit_clfg_orig.py   <- ORIGINAL, untouched. Used by verify to compare.
    fiducial_parameters.py <- reconstructed cosmology constants. REPLACE with yours.
    profile_clfg.py        <- run this to profile (uses ./dummy_ps.h5)
    verify_clfg.py         <- checks optimized vs original give identical output
    make_dummy_clfg.py     <- regenerates dummy_ps.h5
    dummy_ps.h5            <- fake INPUT power-spectrum file (shapes matched)
    frb/ pymaster/ healpy/ <- stubs (you likely have the real ones; harmless)

## Setup
    pip install numpy scipy astropy h5py emcee corner tqdm camb nestle dynesty

## Run the profile
    python profile_clfg.py
Prints top-30 by tottime; saves clfg_profile.prof.

## Verify optimizations didn't change the science
    python verify_clfg.py
Expect: identical: True   (max abs diff 0.0)

## When you get the REAL frbxgal input file
Point PS_FILE in profile_clfg.py at it. The real file must contain datasets:
bin_edges, bin_centers, lmax, n_dm_b, n_z_b, cl, error.
(NOTE: an mcmc_out.h5 is OUTPUT, not input -- it will NOT work here.)

## Optimizations applied so far (both verified identical)
 1. get_n_z_norm() memoized on (ALPHA, Z_STAR): the incomp_gamma chain ran
    50x per likelihood eval with identical args. [247s -> 178s]
 2. model_cl_fg(): the (mu_h,sig_h,ALPHA,Z_STAR)-only block -- including 3
    np.gradient calls and the astropy unit conversion -- cached so it runs
    once per eval instead of 50x. [178s -> 99s]
Remaining targets: interp1d->np.interp, get_cpspec zero-skip, cap_P_vector.
