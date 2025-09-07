import torch
import spintorch

class Gaussian2DWaveSource(spintorch.WaveSource): # also includes the parent class spintorch.WaveSource
    # we want to make sure, that spintorch sees our class as the WaveSource too
    def __init__(self, nx, ny, dx, dy, x0_m, y0_m, sigma_x_m, sigma_y_m, dim=2, prune=1e-3):
        # convert SI → grid indices
        X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing='ij') # kinda like in mumax3
        X_m = X * dx; Y_m = Y * dy

        W = torch.exp(-((X_m - x0_m)**2)/(2*sigma_x_m**2) - ((Y_m - y0_m)**2)/(2*sigma_y_m**2)) # Gaussian
        idx = W > W.max()*prune 
        xi, yi = torch.where(idx) # cutoff everything lower tha 1e-3
        w = W[xi, yi]; w = w / w.sum() # normalization 

        super().__init__(xi, yi, dim) # parent class constructor — spintorch.WaveSource.__init__
        self.register_buffer('weights', w) 
        # register_buffer makes weights non-trainable. So as written,
        # there are no parameters at all — meaning model.parameters() is empty and optimizer.step() does nothing
        """
        register_buffer is a PyTorch nn.Module method.
        It attaches a tensor (w) to the module with a name ("weights").

        Buffers are:
        Stored as part of the models state (state_dict) → they get saved/loaded with the model.
        Moved with the model when you do .to(device) (e.g., CPU → GPU).
        Not considered model parameters (so not updated by optimizers during training).
        Here, w contains the Gaussian weight distribution for the wave source, normalized.
        """

    def forward(self, B, Bt): # class method that lets us update the external field?
        B = B.clone()
        B[self.dim, self.x, self.y] += Bt * self.weights
        return B

def build_sources(cfg, nx, ny, dx, dy):
    # first source at mid-y   [-x-]
    srcs = []
    x0_m = cfg["source"]["x0_m"] # place the first source on the x-axis
    sigma_x_m = cfg["source"]["sigma_x_m"]
    sigma_y_m = cfg["source"]["sigma_y_m"]

    srcs.append(Gaussian2DWaveSource(nx, ny, dx, dy, x0_m, ny*dy/2, # it is placed in the middle of the y - axis
                                     sigma_x_m, sigma_y_m)) # maybe we want to change it to be more flexible
 
    # additional ports defined fractionally along y [o-o-o-o-o-x-o-o-o-o-o]
    for fy in cfg["source"]["ports_fractional_y"]:
        y0_m = fy * ny * dy
        srcs.append(Gaussian2DWaveSource(nx, ny, dx, dy, x0_m, y0_m,
                                         sigma_x_m, sigma_y_m))
    return srcs

def temporal_envelope(cfg, dt, timesteps, device):
    t = torch.arange(0, timesteps*dt, dt, device=device).unsqueeze(0).unsqueeze(2) # does not include timesteps*dt 
    # PyTorch broadcasting rules often expect tensors with extra singleton dimensions.
    Bt = cfg["source"]["Bt_T"] # external field magnitude
    temp = cfg["source"]["temporal"] # envelope parameters
    if temp["type"] == "gaussian":
        tc = temp["center_frac"] * dt * timesteps
        s  = temp["sigma_frac"]  * dt * timesteps
        env = torch.exp(-((t - tc)**2)/(2*s**2))
        return Bt * env, t
    raise ValueError("Unknown temporal type")
