import os, torch
import numpy as np
import matplotlib.pyplot as plt
import spintorch
from spintorch.plot import geometry as plot_geometry, wave_integrated, wave_snapshot

def solve_once(model, inputs, savedir, plotdir, cfg, tag="run"):
    os.makedirs(savedir, exist_ok=True); os.makedirs(plotdir, exist_ok=True)

    # freeze any accidental trainables (esp. geom.rho)
    if hasattr(model, "geom") and hasattr(model.geom, "rho"):
        model.geom.rho.requires_grad_(False)
    model.eval()

    # run forward without grads
    with torch.no_grad():
        model.retain_history = bool(cfg.get("save_history", True))
        u = model(inputs).sum(dim=1)   # [batch=1, Nprobes]
        np.save(os.path.join(savedir, f"outputs_{tag}.npy"), u.detach().cpu().numpy())

    # lightweight geometry plot
    plot_geometry(model, epoch=0, plotdir=plotdir)

    # optional history plots
    if model.retain_history and cfg.get("plots", {}).get("snapshots", True):
        T = cfg["time"]["timesteps"]
        mz = torch.stack(model.m_history, 1)[0, :, 2, ] - model.m0[0, 2, ].unsqueeze(0).cpu()
        wave_snapshot(model, mz[T-1], os.path.join(plotdir, f"snapshot_t{T}_{tag}.png"), r"$m_z$")
        wave_snapshot(model, mz[max(T//2-1,0)], os.path.join(plotdir, f"snapshot_t{max(T//2,1)}_{tag}.png"), r"$m_z$")
        if cfg.get("plots", {}).get("integrated", True):
            wave_integrated(model, mz, os.path.join(plotdir, f"integrated_{tag}.png"))

    return u
