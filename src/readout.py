import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union

Tensor = torch.Tensor

def _roi_mask_from_box(
    nx:int, ny:int, dx:float, dy:float,
    box_xy: Optional[Tuple[float,float,float,float]] = None,
    box_idx: Optional[Tuple[int,int,int,int]] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """
    Build a binary mask [nx, ny] for a rectangular ROI (Region of Interest)
      - box_xy = (x_min_m, x_max_m, y_min_m, y_max_m) in meters
      - OR box_idx = (xi_min, xi_max, yi_min, yi_max) in indices (inclusive:exclusive)
    """
    if (box_xy is None) == (box_idx is None):
        raise ValueError("Provide exactly one of box_xy or box_idx.")

    if box_idx is None:
        x_min_m, x_max_m, y_min_m, y_max_m = box_xy
        xi_min = max(0, int(x_min_m / dx))
        xi_max = min(nx, int(torch.ceil(torch.tensor(x_max_m / dx)).item())) # to make sure that the border is included
        yi_min = max(0, int(y_min_m / dy))
        yi_max = min(ny, int(torch.ceil(torch.tensor(y_max_m / dy)).item()))
    else:
        xi_min, xi_max, yi_min, yi_max = box_idx
        xi_min = max(0, xi_min); xi_max = min(nx, xi_max)
        yi_min = max(0, yi_min); yi_max = min(ny, yi_max)

    mask = torch.zeros((nx, ny), dtype=dtype, device=device) # mask shaped liek the whole simulation area
    if xi_max > xi_min and yi_max > yi_min:
        mask[xi_min:xi_max, yi_min:yi_max] = 1.0 # here we only "cut out" the ROI section 
    return mask


class ROIIntegrator(nn.Module):
    """
    Time-integrated intensity over ROI:
      I = ∫_{t1}^{t2} <m_x^2 + m_y^2>_ROI dt
    Inputs:
      m: [B, T, 3, nx, ny]
    Returns:
      out: [B] (scalar per batch item)
    """
    def __init__(
        self,
        nx:int, ny:int, dx:float, dy:float,
        t_start_idx:int, t_end_idx:int,
        dt_s: float,
        box_xy: Optional[Tuple[float,float,float,float]] = None,
        box_idx: Optional[Tuple[int,int,int,int]] = None,
        normalize_area: bool = True,
        mx_idx:int = 0, my_idx:int = 1,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.t0 = int(t_start_idx)
        self.t1 = int(t_end_idx)
        self.dt = float(dt_s)
        self.normalize_area = bool(normalize_area)
        self.mx_idx = int(mx_idx)
        self.my_idx = int(my_idx)

        mask = _roi_mask_from_box(nx, ny, dx, dy, box_xy=box_xy, box_idx=box_idx, device=device, dtype=dtype)
        # [1,1,1,nx,ny] for broadcasting with [B,T,3,nx,ny]
        self.register_buffer("mask", mask.view(1,1,1,nx,ny))
        area = mask.sum()
        self.register_buffer("area", area if area > 0 else torch.tensor(1.0, device=device, dtype=dtype))

    def forward(self, m: Tensor) -> Tensor:
        """
        m: [B, T, 3, nx, ny]
        """
        assert m.ndim == 5 and m.shape[2] >= 2, "Expected m shape [B,T,3,nx,ny] with at least x,y components"
        # time crop
        m_seg = m[:, self.t0:self.t1]                      # [B, T', 3, nx, ny]
        # select x,y components
        mx = m_seg[:, :, self.mx_idx:self.mx_idx+1]       # [B, T', 1, nx, ny]
        my = m_seg[:, :, self.my_idx:self.my_idx+1]       # [B, T', 1, nx, ny]
        power = (mx*mx + my*my)                           # [B, T', 1, nx, ny]
        # apply ROI mask
        roi_power = power * self.mask                     # broadcast
        # spatial average (optionally)
        if self.normalize_area:
            roi_power = roi_power.sum(dim=(-1,-2)) / (self.area + 1e-20)  # [B, T', 1]
        else:
            roi_power = roi_power.sum(dim=(-1,-2))                          # [B, T', 1]
        # integrate over time
        out = roi_power.sum(dim=1).squeeze(-1) * self.dt   # [B]
        return out


class BandpassROI(nn.Module):
    """
    Bandpassed energy around f0 using a short-time FFT (Gaussian frequency window),
    then averaged over ROI.

    Steps (all differentiable):
      - take m_x, m_y over ROI, spatially average each frame
      - (optional) apply Hann window in time
      - rFFT over time → frequency bins
      - apply Gaussian frequency weights centered at f0 with std 'bw_Hz'
      - energy = sum |FFT|^2 * weights  (for both x,y), scalar per batch

    Inputs:
      m: [B, T, 3, nx, ny]
    Returns:
      out: [B]
    """
    def __init__(
        self,
        nx:int, ny:int, dx:float, dy:float,
        dt_s: float,
        f0_Hz: float,
        bw_Hz: float,
        box_xy: Optional[Tuple[float,float,float,float]] = None,
        box_idx: Optional[Tuple[int,int,int,int]] = None,
        use_hann_time: bool = True,
        mx_idx:int = 0, my_idx:int = 1,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.dt = float(dt_s)
        self.f0 = float(f0_Hz)
        self.bw = float(bw_Hz)
        self.use_hann_time = bool(use_hann_time)
        self.mx_idx = int(mx_idx)
        self.my_idx = int(my_idx)

        mask = _roi_mask_from_box(nx, ny, dx, dy, box_xy=box_xy, box_idx=box_idx, device=device, dtype=dtype)
        self.register_buffer("mask2d", mask)  # [nx, ny]
        area = mask.sum()
        self.register_buffer("area", area if area > 0 else torch.tensor(1.0, device=device, dtype=dtype))

        # place-holders; actual frequency window depends on T at forward()
        self.register_buffer("last_freqs", torch.zeros(1, dtype=dtype, device=device), persistent=False)
        self.register_buffer("last_wfreq", torch.zeros(1, dtype=dtype, device=device), persistent=False)
        self.register_buffer("last_hann", torch.zeros(1, dtype=dtype, device=device), persistent=False)

    def _ensure_windows(self, T:int, device, dtype):
        # build Hann window in time if requested
        if self.use_hann_time:
            n = torch.arange(T, device=device, dtype=dtype)
            hann = 0.5 - 0.5*torch.cos(2*torch.pi*n/(T-1))
        else:
            hann = torch.ones(T, device=device, dtype=dtype)
        # rFFT frequency bins
        freqs = torch.fft.rfftfreq(T, d=self.dt).to(device=device, dtype=dtype)  # [Nf]
        # Gaussian frequency weights centered at f0
        wfreq = torch.exp(-0.5*((freqs - self.f0)/ (self.bw + 1e-30))**2)

        self.last_freqs = freqs
        self.last_wfreq = wfreq
        self.last_hann = hann

    def forward(self, m: Tensor) -> Tensor:
        """
        m: [B, T, 3, nx, ny]
        """
        assert m.ndim == 5 and m.shape[2] >= 2, "Expected m shape [B,T,3,nx,ny]"
        B, T, _, nx, ny = m.shape
        device = m.device; dtype = m.dtype

        # build windows for this T
        self._ensure_windows(T, device, dtype)

        # spatial averaging over ROI (apply mask and divide by area)
        mask = self.mask2d.to(device=device, dtype=dtype)          # [nx, ny]
        mx = m[:, :, self.mx_idx] * mask                           # [B, T, nx, ny]
        my = m[:, :, self.my_idx] * mask
        mx = mx.sum(dim=(-1,-2)) / (self.area.to(device=device, dtype=dtype) + 1e-20)  # [B, T]
        my = my.sum(dim=(-1,-2)) / (self.area.to(device=device, dtype=dtype) + 1e-20)  # [B, T]

        # apply time window
        w = self.last_hann.view(1, T)                              # [1, T]
        mxw = mx * w                                               # [B, T]
        myw = my * w

        # rFFT over time
        MX = torch.fft.rfft(mxw, dim=1)                            # [B, Nf]
        MY = torch.fft.rfft(myw, dim=1)

        # Gaussian frequency weights
        Wf = self.last_wfreq.view(1, -1)                           # [1, Nf]
        # energy accumulation in frequency domain (sum |F|^2 * Wf)
        Ex = (MX.abs()**2 * Wf).sum(dim=1)                         # [B]
        Ey = (MY.abs()**2 * Wf).sum(dim=1)                         # [B]

        # Optionally scale by df to approximate integral in frequency
        df = (self.last_freqs[1] - self.last_freqs[0]) if self.last_freqs.numel() > 1 else torch.tensor(1.0, device=device, dtype=dtype)
        out = (Ex + Ey) * df                                       # [B]
        return out
