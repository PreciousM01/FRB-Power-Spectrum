"""
ryan_cl.py -- Ryan's DM-galaxy cross spectrum model (Dxg / Cl), extracted from
Dxg_theory.ipynb. Includes everything Dxg() needs: Friedmann distances, the CAMB
nonlinear power spectrum, the tracer spectra (Peg, Pfg), the FRB distribution
n_f_dist, the background DM D_of_chi, the host-DM mean, and the Cl beam.

Plotting cells omitted. Imports cleanly; CAMB runs once at import (a few seconds).

Key callables:
    Dxg(l, z_g, ...)            -> term1+term2 (or parts) of the cross spectrum
    cl_model(ell, dxg, l_cut)   -> applies Gaussian beam suppression
    n_f_dist(chi, zstar, alpha) -> FRB radial distribution
"""
import numpy as np
import scipy
from scipy import interpolate
from scipy import integrate
from scipy.integrate import simpson, cumulative_trapezoid
from scipy import special as sps
import camb
from camb import model

# ---- survey config (Ryan's defaults) ----
ZSTAR = 1.0
ALPHA = 0.1
B_E = 1.0
B_F = 1.5
LOG_K_CUT = np.log(1.65)
MU_HOST = 100.0

# ---- cosmology / Friedmann distances ----
HUBBLE_DISTANCE = 2997.92458
OMEGA_M = 0.3
OMEGA_R = 4.15e-5
OMEGA_L = 1 - OMEGA_M - OMEGA_R
LITTLE_H = 0.6770

def dtH0_da(a):
    return 1. / np.sqrt(OMEGA_R / a**2 + OMEGA_M / a + OMEGA_L * a**2)
def detaH0_da(a):
    return dtH0_da(a) / a

def solve_friedmann(return_extra=False):
    a = np.logspace(-3., 0, 5000, endpoint=True)
    da = np.diff(a); a_c = a[1:] - da/2; a = a[1:]; z = 1/a - 1
    etaH0 = np.cumsum(detaH0_da(a_c) * da)
    conformal_distance = (etaH0[-1] - etaH0) * HUBBLE_DISTANCE
    if return_extra:
        dz_dchi = np.gradient(z, conformal_distance)
        return (interpolate.interp1d(z, conformal_distance, kind='cubic'),
                interpolate.interp1d(conformal_distance, z, kind='cubic'),
                interpolate.interp1d(z, dz_dchi, kind='cubic'))
    return interpolate.interp1d(z, conformal_distance, kind='cubic')

chi_of_z, z_of_chi, dzdchi = solve_friedmann(return_extra=True)

# ---- CAMB nonlinear power spectrum (Ryan's setup) ----
h = 0.6770
ombh2 = 0.0486 * h**2
omch2 = (0.3089 - 0.0486) * h**2
z_pk = np.linspace(0, 5, 100)
pars = camb.CAMBparams()
pars.set_cosmology(H0=100.0*h, ombh2=ombh2, omch2=omch2, mnu=0.06, tau=0.0568)
pars.InitPower.set_params(As=2.107e-9, ns=0.9682)
pars.set_matter_power(redshifts=z_pk, kmax=50.0)
pars.NonLinear = model.NonLinear_both
results = camb.get_results(pars)
kh, z_eval, pk_table = results.get_matter_power_spectrum(minkh=1e-4, maxkh=40.0, npoints=200)
pk_interp = scipy.interpolate.RegularGridInterpolator((z_eval, kh), pk_table)
z_lo, z_hi = z_eval[0], z_eval[-1]
k_lo, k_hi = kh[0], kh[-1]

def Pk_NL(z, k):
    k_arr = np.atleast_1d(k)
    z_arr = np.broadcast_to(np.atleast_1d(z), k_arr.shape)
    k_arr = np.clip(k_arr, k_lo, k_hi)
    z_arr = np.clip(z_arr, z_lo, z_hi)
    out = pk_interp(np.column_stack([z_arr, k_arr]))
    if np.ndim(k) == 0 and np.ndim(z) == 0:
        return out[0]
    return out.reshape(np.shape(k) if np.ndim(k) else np.shape(z))

b_g = 1.2

def Peg(z, k, b_e=B_E, log_k_cut=LOG_K_CUT):
    return Pk_NL(z, k) * b_e * np.exp(-k/np.exp(log_k_cut)) * b_g
def Pfg(z, k, b_f=B_F):
    return Pk_NL(z, k) * b_f * b_g

# ---- distribution / DM background ----
n_e0 = 0.24203
DIST_MIN, DIST_MAX, N_INT = 50, 6e3, 150

def _upper_incgamma(a, x):
    if a > 0:
        return sps.gammaincc(a, x) * sps.gamma(a)
    if np.isclose(a, 0.0):
        return -sps.expi(-x)
    if (-1.0 < a) and (a < 0.0):
        return (_upper_incgamma(a+1.0, x) - np.exp(-x)*x**a) / a
    eps = 1e-6
    return sps.gammaincc(a+eps, x) * sps.gamma(a+eps)

def _schechter_cumulative_above(alpha, x_min):
    return _upper_incgamma(alpha + 1.0, x_min)

_z_grid = np.linspace(0.01, 6, 1000)
_chi_grid = chi_of_z(_z_grid)
_dl_grid = (1 + _z_grid) * _chi_grid
_dchi_dz = 1/dzdchi(_z_grid)
_dl_of_z = interpolate.interp1d(_z_grid, _dl_grid, kind='cubic')

def _n_z_norm(zstar, alpha):
    dL_star = float(_dl_of_z(zstar))
    x_min = (_dl_grid / dL_star)**2
    n_un = _schechter_cumulative_above(alpha, x_min)
    denom = simpson(4.0*np.pi*_chi_grid**2 * n_un * _dchi_dz, _z_grid)
    return n_un / denom

_CACHE = {}
def n_f_dist(chi, zstar=ZSTAR, alpha=ALPHA):
    chi = np.asarray(chi, dtype=float)
    z = z_of_chi(chi)
    key = f"{zstar}_{alpha}"
    if key not in _CACHE:
        _CACHE[key] = _n_z_norm(zstar, alpha)
    n_of_z = np.interp(z, _z_grid, _CACHE[key], left=0.0, right=0.0)
    return 4.0*np.pi*chi**2*n_of_z

# background DM D(chi) = int (1+z) n_e0 dchi
_chi_vals = np.linspace(0, 6000, 1000)
_z_vals = z_of_chi(_chi_vals)
_D_vals = cumulative_trapezoid((1 + _z_vals)*n_e0, _chi_vals, initial=0)
D_of_chi = interpolate.interp1d(_chi_vals, _D_vals, kind='cubic')

def mu_host_func(chi, muR=MU_HOST, disable_z_evolve=False):
    if disable_z_evolve:
        return muR
    return muR/(1 + z_of_chi(chi))

# FRB-weighted mean DM (the D_f subtraction term)
_chi_int = np.geomspace(DIST_MIN, DIST_MAX, N_INT)
FRB_WEIGHT_DM = integrate.simpson(
    n_f_dist(_chi_int) * (D_of_chi(_chi_int) + mu_host_func(_chi_int)), _chi_int)

# ---- the cross spectrum ----
def Dxg(l, z_g, zstar=ZSTAR, alpha=ALPHA, b_e=B_E, b_f=B_F,
        log_k_cut=LOG_K_CUT, mu_host=MU_HOST, D_f_external=None, return_parts=False):
    chi_g = chi_of_z(z_g)
    prefactor1 = n_e0*(1+z_g)*Peg(z=z_g, k=l/chi_g, b_e=b_e, log_k_cut=log_k_cut)/chi_g**2
    chi_vals_gmax = np.linspace(chi_g, DIST_MAX, N_INT)
    term1 = prefactor1 * integrate.simpson(n_f_dist(chi_vals_gmax, zstar, alpha), chi_vals_gmax)
    prefactor2 = n_f_dist(chi_g, zstar, alpha)*Pfg(z=z_g, k=l/chi_g, b_f=b_f)/chi_g**2
    D_f = FRB_WEIGHT_DM if D_f_external is None else float(D_f_external)
    term2 = prefactor2*(D_of_chi(chi_g) + mu_host_func(chi_g, muR=mu_host) - D_f)
    if return_parts:
        return term1, term2
    return term1 + term2

def cl_model(ell, dxg, l_cut=4000):
    return dxg * np.exp(-ell**2 / (2*l_cut**2))
