import torch, os
import spintorch

def _raster_scatter_mask(nx, ny, dx, dy, cfg_scatt, device, dtype):
    """
    Returns a [nx, ny] binary mask with 1's inside circular scatterers.
    Supports modes: 'single' or 'grid'.
    """
    mode      = cfg_scatt.get("mode", "single")
    diam_m    = float(cfg_scatt.get("diameter_m", 150e-9))
    r_cells_x = max(1, int(round((diam_m/2) / dx)))
    r_cells_y = max(1, int(round((diam_m/2) / dy)))

    x = torch.arange(nx, device=device, dtype=dtype).view(-1, 1)
    y = torch.arange(ny, device=device, dtype=dtype).view(1, -1)
    mask = torch.zeros((nx, ny), device=device, dtype=dtype)

    if mode == "single":
        x0_m, y0_m = cfg_scatt.get("single_center_m", [nx*dx/2, ny*dy/2])
        cx, cy = int(round(x0_m/dx)), int(round(y0_m/dy))
        m = (x - cx)**2 / (r_cells_x**2) + (y - cy)**2 / (r_cells_y**2) <= 1.0
        mask[m] = 1.0
        return mask

    # grid mode
    pitch_m = float(cfg_scatt.get("pitch_m", 400e-9))
    px = max(1, int(round(pitch_m/dx)))
    py = max(1, int(round(pitch_m/dy)))
    span_x_m = float(cfg_scatt.get("nx_span_m", nx*dx))
    span_y_m = float(cfg_scatt.get("ny_span_m", ny*dy))
    span_x = max(1, int(round(span_x_m/dx)))
    span_y = max(1, int(round(span_y_m/dy)))

    # center the grid in the film
    x_start = cfg_scatt.get("scattering_mask_shift")[0]#max(, nx//2 - span_x//2)
    y_start = cfg_scatt.get("scattering_mask_shift")[1]#max(, ny//2 - span_y//2)
    x_centers = list(range(x_start + r_cells_x, min(nx - r_cells_x, x_start + span_x), px))
    y_centers = list(range(y_start + r_cells_y, min(ny - r_cells_y, y_start + span_y), py))

    for cx in x_centers:
        for cy in y_centers:
            m = (x - cx)**2 / (r_cells_x**2) + (y - cy)**2 / (r_cells_y**2) <= 1.0
            mask[m] = 1.0
    return mask

def make_wavegeom(cfg):
    """
    Build a WaveGeometryMs; if cfg['scatterers']['enabled'] is True,
    emulate PMA dots by locally scaling Ms via geom.rho (Ms map).
    """
    nx, ny = cfg["grid"]["nx"], cfg["grid"]["ny"]
    dx, dy, dz = cfg["grid"]["dx_m"], cfg["grid"]["dy_m"], cfg["grid"]["dz_m"]
    Ms  = float(cfg["material"]["Ms_Aperm"])
    B0  = float(cfg["bias"]["B0_T"])

    # Base uniform film
    geom = spintorch.WaveGeometryMs((nx, ny), (dx, dy, dz), Ms, B0)

    # If no scatterers requested, return as-is
    scatt = cfg.get("scatterers", {})
    if not scatt or not scatt.get("enabled", False):
        return geom

    # Build binary mask for scatterers (1 inside dots)
    device = geom.rho.device if hasattr(geom, "rho") else torch.device("cpu")
    dtype  = geom.rho.dtype  if hasattr(geom, "rho") else torch.float32
    mask = _raster_scatter_mask(nx, ny, dx, dy, scatt, device, dtype)

    # Scale Ms inside dots; outside remains baseline Ms.
    Ms_scale = float(scatt.get("Ms_scale", 0.7))  # e.g., 0.7*Ms inside dots
    rho = torch.ones((nx, ny), device=device, dtype=dtype)
    rho = torch.where(mask > 0, torch.tensor(Ms_scale, device=device, dtype=dtype), rho)

    # In SpinTorch, effective Ms map is Ms0 * rho. Set and (optionally) freeze it.
    with torch.no_grad():
        geom.rho.copy_(rho)  # geom.rho exists and is [nx, ny]
    # For solver-only runs ensure it's not trainable:
    geom.rho.requires_grad_(False)

    return geom