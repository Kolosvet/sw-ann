import os, torch
from spintorch.plot import plot_output, plot_loss, geometry as plot_geometry, wave_integrated, wave_snapshot

def my_loss(output, target_index):
    # maximize intensity at target_index (desired output) relative to others
    target_value = output[:, target_index] # "desired" outputs for each sample  - column with target index
    loss = output.sum(dim=1)/target_value - 1 # wrong/all
    # log for scale stability
    return (loss.sum()/loss.size(0)).log10() # loss.mean()

def train_focus(model, inputs, outputs_idx, cfg, savedir, plotdir, nepoch=20, retain_history=True):

    # make directories for output 
    os.makedirs(savedir, exist_ok=True); os.makedirs(plotdir, exist_ok=True)

    # pick the optimizer
    # this if we ddon't want source optimization
    model.geom.rho.requires_grad_(False)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    # # this if we want source optimization
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_iter = []

    dt = cfg["time"]["dt_s"]; T = cfg["time"]["timesteps"]

# PyTorch computes the gradient of the loss with respect to all model parameters and accumulates them in each parameter’s .grad attribute.
    for epoch in range(nepoch):
        optimizer.zero_grad() # clear old gradients
        u = model(inputs).sum(dim=1)   # [batch=1, probes] forward
        plot_output(u[0,], outputs_idx+1, epoch, plotdir)
        loss = my_loss(u, outputs_idx) # pick target probe index
        loss_iter.append(loss.item()) # remember the loss 
        plot_loss(loss_iter, plotdir)
        loss.backward() # backpropagation 
        optimizer.step() 
        print(f"Epoch {epoch:02d}  Loss {loss.item():.6f}")

        torch.save({
            'epoch': epoch, 'loss_iter': loss_iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(savedir, f"model_e{epoch}.pt"))

        if retain_history and hasattr(model, "m_history") and len(model.m_history)>0:
            with torch.no_grad():
                plot_geometry(model, epoch=epoch, plotdir=plotdir)
                mz = torch.stack(model.m_history, 1)[0, :, 2, ] - model.m0[0, 2, ].unsqueeze(0).cpu()
                wave_snapshot(model, mz[T-1], os.path.join(plotdir, f"snapshot_t{T}_e{epoch}.png"), r"$m_z$")
                wave_snapshot(model, mz[int(T/2)-1], os.path.join(plotdir, f"snapshot_t{int(T/2)}_e{epoch}.png"), r"$m_z$")
                wave_integrated(model, mz, os.path.join(plotdir, f"integrated_e{epoch}.png"))
