"""
test_mcmcfit_dxg.py -- regression test suite for the optimized cl_Dg model.

Run from inside profiling_kit:
    pytest test_mcmcfit_dxg.py -v

Self-contained: uses ./dummy_data. No real catalog needed.
Requires mcmcfit_orig.py (frozen original) alongside mcmcfit.py for the
optimized-vs-original test. If you don't have mcmcfit_orig.py, get it from git:
    git show HEAD:path/to/mcmcfit.py > mcmcfit_orig.py
"""
import numpy as np
import h5py
import glob, os
import pytest
import scipy.special as sps

import mcmcfit as mf
try:
    import mcmcfit_orig as mf_orig
    HAVE_ORIG = True
except ImportError:
    HAVE_ORIG = False

DATA_DIR = "../../dxg_measurement"               
ZS = [0.075, 0.15, 0.25, 0.35, 0.45]
# 6 params: DM_H_bar, cut_scale, l_cut, bf0, ALPHA, Z_STAR
THETA = (60.0, -1.0, 4000.0, 1.5, -1.5, 1.0)

def _build(module):
    files = sorted(glob.glob(os.path.join(DATA_DIR, "dmxgal_*_combined.h5")))
    assert len(files) == 5, f"expected 5, found {len(files)}: {files}"
    # no real FRB catalog -> stand-in dm_select. Only affects Nf/D_bar, which are
    # identical for optimized and original, so the equivalence test stays valid.
    dm_select = np.random.default_rng(0).normal(500, 200, 500)
    obj = module.cl_Dg_fit(dm_select)
    obj.read_data(files, files)
    return obj

@pytest.fixture(scope="module")
def g():
    return _build(mf)

@pytest.fixture(scope="module")
def g_orig():
    if not HAVE_ORIG:
        pytest.skip("mcmcfit_orig.py not present")
    return _build(mf_orig)


def test_incomp_gamma_matches_scipy(g):
    for a, x in [(2.0, 1.0), (1.5, 0.3), (3.0, np.array([0.1, 1.0, 5.0]))]:
        ref = sps.gammaincc(a, x) * sps.gamma(a)
        assert np.allclose(g.incomp_gamma(a, x), ref, rtol=1e-12)

def test_pf_normalized(g):
    # get_p_z_norm returns (pf_z, n); pf_chi = 4 pi chi^2 n/Nf integrates to ~1 over chi
    pf_z, n = g.get_p_z_norm(-1.5, 1.0)
    from scipy.integrate import trapezoid
    pf_chi = 4*np.pi*g.chi_grid**2*n/g.Nf
    area = trapezoid(pf_chi, g.chi_grid)
    assert abs(area - 1.0) < 1e-2

def test_cache_matches_fresh(g):
    fresh = _build(mf)
    cached = g.get_p_z_norm(-1.5, 1.0)[0]
    cold = fresh.get_p_z_norm(-1.5, 1.0)[0]
    assert np.allclose(cached, cold, rtol=1e-12)

def test_cache_different_params_differ(g):
    a = g.get_p_z_norm(-1.5, 1.0)[0]
    b = g.get_p_z_norm(-0.5, 0.7)[0]
    assert not np.allclose(a, b)

def test_combined_model_finite_and_shape(g):
    g.convolve = False
    out = g.combined_model(*THETA)
    assert np.all(np.isfinite(out))
    # 5 z bins x (14-4) l bins = 50
    assert out.shape == (50,)

def test_model_cl_Dg_plot_parts_sum(g):
    # model_cl_Dg_plot returns (total, term1, term2); total should equal t1+t2
    # (up to the beam, which is applied to all three equally)
    l = np.linspace(1, g.lmax, g.lmax)
    total, t1, t2 = g.model_cl_Dg_plot(l, 2, *THETA)
    assert np.allclose(total, t1 + t2, rtol=1e-10)

@pytest.mark.skipif(not HAVE_ORIG, reason="mcmcfit_orig.py not present")
def test_optimized_matches_original(g, g_orig):
    g.convolve = False
    g_orig.convolve = False
    out_opt = g.combined_model(*THETA)
    out_org = g_orig.combined_model(*THETA)
    assert np.allclose(out_opt, out_org, rtol=1e-12, atol=0.0), \
        f"max diff {np.max(np.abs(out_opt - out_org))}"

@pytest.mark.skipif(not HAVE_ORIG, reason="mcmcfit_orig.py not present")
@pytest.mark.parametrize("theta", [
    (40.0, -0.5, 2000.0, 1.2, -1.8, 0.5),
    (90.0, 0.5, 6000.0, 2.5, 0.5, 1.4),
])
def test_optimized_matches_original_multi(g, g_orig, theta):
    g.convolve = False
    g_orig.convolve = False
    assert np.allclose(g.combined_model(*theta), g_orig.combined_model(*theta),
                       rtol=1e-10, atol=0.0)