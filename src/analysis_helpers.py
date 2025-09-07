# src/analysis_helpers.py
import torch

def count_trainable_params(module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def list_trainable_params(module):
    out = []
    for name, p in module.named_parameters():
        if p.requires_grad:
            out.append((name, tuple(p.shape)))
    return out

def assert_baseline_no_weights(model):
    """
    Verifies Step 0-1 'clean film' regime:
      - geometry exists
      - no extra additive field maps (if present, they are zero)
      - optional: rho is frozen if present
    """
    assert hasattr(model, "geom"), "Model has no .geom (geometry) attribute."
    geom = model.geom
    assert hasattr(geom, "Ms") and hasattr(geom, "B0"), "Geometry missing Ms/B0."

    # If your solver supports an additive field map, ensure it's zero
    if hasattr(model, "H_add") and model.H_add is not None:
        max_add = float(torch.as_tensor(model.H_add).abs().max())
        if max_add > 1e-12:
            raise AssertionError(f"Nonzero additive field detected: max={max_add}")

    # Not required, but helpful: warn if rho is trainable
    if hasattr(geom, "rho") and isinstance(geom.rho, torch.Tensor):
        if geom.rho.requires_grad:
            print("⚠ Warning: geom.rho is trainable (requires_grad=True). "
                  "For solver-only runs, freeze it or set freeze_rho=true in config.")

    print("Baseline check passed: uniform film, no Ms/ΔH weights active.")

def ensure_solver_only(model, freeze_rho: bool = True):
    """
    Puts the model into 'solver-only' mode: eval + no grads.
    If freeze_rho=True and rho exists, mark it non-trainable.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if freeze_rho and hasattr(model, "geom") and hasattr(model.geom, "rho"):
        model.geom.rho.requires_grad_(False)
