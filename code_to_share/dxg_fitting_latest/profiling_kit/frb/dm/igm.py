"""Stub of frb.dm.igm for profiling only.
average_DM(z) approximates the Macquart relation <DM_cosmic> ~ 855*z pc/cm^3.
Exact value is irrelevant for per-call profiling; it is only used once in __init__.
"""
import astropy.units as u

def average_DM(z, **kwargs):
    return (855.0 * float(z)) * u.pc / u.cm**3
