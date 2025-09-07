import json, os

def load_configs(root=".", mats="configs/materials.json", sim="configs/sim_defaults.json"):
    with open(os.path.join(root, mats), "r") as f: # load the material parameters 
        mats_cfg = json.load(f) # dictionary of dictionaries 
    with open(os.path.join(root, sim), "r") as f:
        sim_cfg = json.load(f) # same goes for simualtions parameters 

    mat_name = mats_cfg.get("default") # dictionary keys holds the name of default configuration "YIG_100nm"
    mat = mats_cfg["materials"][mat_name]

    # flatten a convenient dict
    cfg = {
        **sim_cfg, # take all key/value pairs from sim_cfg
        "material": mat, # add/overwrite key "material"
        "material_name": mat_name  # add/overwrite key "material_name"
    }
    return cfg
