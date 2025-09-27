import spintorch

def build_disk_probes(cfg, nx, ny):
    Np = cfg["probes"]["Ndisk"] # number of readouts
    x_index = cfg["probes"]["x_index"] % nx  # allow negative indexing
    r = cfg["probes"]["radius_cells"]
    # it's cool, that spintorch has a function just for this
    probes = [spintorch.WaveIntensityProbeDisk(x_index, int(ny*(p+1)/(Np+1)), r) for p in range(Np)]
    return probes
