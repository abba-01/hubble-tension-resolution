"""
Minimal UHA helpers for CosmoID + address fields.

This is a placeholder to fingerprint the encode-time cosmology and to
demonstrate the (a, ξ, û) tuple generation from (z, RA, Dec) under a prior.
"""
import math
from dataclasses import dataclass

@dataclass
class CosmoID:
    H0: float
    Omega_m: float
    Omega_r: float
    Omega_L: float

def comoving_horizon(a, H0, Om, Or, OL, c=299792458.0):
    # Simplified; in practice use an ODE/integral with high accuracy.
    return c / H0  # placeholder

def uha_from_sky(ra_deg, dec_deg, z, cosmo: CosmoID):
    a = 1.0/(1.0+z)
    # unit vector
    ra = math.radians(ra_deg); dec = math.radians(dec_deg)
    ux = math.cos(dec)*math.cos(ra)
    uy = math.cos(dec)*math.sin(ra)
    uz = math.sin(dec)
    # horizon-normalized radial code (toy)
    RH = comoving_horizon(a, cosmo.H0*1000/3.0856775814913673e22, cosmo.Omega_m, cosmo.Omega_r, cosmo.Omega_L)
    xi = 0.5  # placeholder
    return dict(a=a, xi=xi, u=(ux,uy,uz), CosmoID=cosmo.__dict__)
