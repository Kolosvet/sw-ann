import torch, os
import spintorch

def make_wavegeom(cfg):
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"] # number of grid elements - only 1 element along z
    dx, dy, dz = cfg["grid"]["dx_m"], cfg["grid"]["dy_m"], cfg["grid"]["dz_m"] # ell size
    Ms = cfg["material"]["Ms_Aperm"] # saturation magnetization in the whole film
    B0 = cfg["bias"]["B0_T"] # external field
    geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0) # define spintorch geometry
    return geom
