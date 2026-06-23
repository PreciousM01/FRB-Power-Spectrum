"""Stub of frb.dm.igm for profiling cl_fg_fit (no real data needed).

average_DM(z) approximates the Macquart relation <DM_cosmic> ~ 855*z pc/cm^3.
Supports cumul=True: returns (DM_cumulative_array, z_grid) like the real API,
where DM_cumulative is the running mean DM out to each z on the grid.
Exact values are irrelevant for per-call profiling; this only runs once in __init__.
"""
import numpy as np
import astropy.units as u

_SLOPE = 855.0  # pc cm^-3 per unit z, approximate <DM_cosmic> slope

def average_DM(z, cumul=False, **kwargs):
    if cumul:
        n = 1000
        z_grid = np.linspace(z/n, z, n)        # avoid 0
        dm_cumul = _SLOPE * z_grid             # cumulative mean DM out to each z
        return dm_cumul * u.pc / u.cm**3, z_grid
    return (_SLOPE * float(z)) * u.pc / u.cm**3
