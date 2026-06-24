# Profiling kit for mcmcfit.py

This lets you profile the MCMC **without your real data**. It contains stand-ins
for everything the code needs, so you can see how the ORIGINAL code performs,
find the bottlenecks yourself, then optimize.

## What's in here

    mcmcfit.py              <- your ORIGINAL (unoptimized) code. Profile this first.
    fiducial_parameters.py  <- reconstructed cosmology constants (Planck-like).
                               REPLACE with your real one when you have it.
    profile_mcmc.py         <- the profiling driver. Run this.
    dummy_data/             <- fake FRB catalog + 5 fit + 5 plot .h5 files.
    frb/ pymaster/ healpy/  <- lightweight STUBS. The real packages aren't needed
                               for the non-convolve sampling path we profile.
                               (frb.dm.igm is stubbed with the Macquart relation
                                ~855*z; pymaster/healpy are empty.)

## One-time setup

Use a Python env with the basics installed:

    pip install numpy scipy astropy h5py emcee corner tqdm camb

(camb is the only heavy one. The frb/pymaster/healpy stubs in this folder mean
you do NOT need to install those.)

## Run the profile (Step 1: see the baseline)

    cd profiling_kit
    python profile_mcmc.py

Takes ~1 minute. It prints the 30 slowest functions. Read the column
`tottime` = seconds spent INSIDE that function (not its sub-calls). The
function at the top is your bottleneck.

It also writes `mcmc_profile.prof`. Re-examine it anytime without rerunning:

    python -c "import pstats; s=pstats.Stats('mcmc_profile.prof'); s.sort_stats('cumtime'); s.print_stats(30)"

(`cumtime` = time including sub-calls; good for seeing which high-level
function owns the most total time. `tottime` = the actual hot spot.)

## How to read it / what to optimize (Step 2)

Look for a single function with a large `tottime` and high `ncalls`. That's
where to focus. For each candidate ask:
  - Is it called more times than necessary? (cache / hoist out of loops)
  - Is it doing slow per-call work? (e.g. building objects, unit conversions)

## Verify before/after (Step 3)

After you change something, confirm you didn't change the science:
compare combined_model() output before vs after on the same parameters.
They should match to ~1e-12. (verify_optimization.py does this.)
