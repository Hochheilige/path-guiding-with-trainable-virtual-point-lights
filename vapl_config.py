class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__(config)
        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)
            self[key] = value
        self.__dict__ = self 

raw_config = {
    "scene"     : "cornell box", # cornell box, veach mis
    "mode"      : "local",       # local, wandb, sweep
    "sweep_config" : None,       # add to use sweep
    "run_name"     : None,       # for wandb runs
    "grid" : {
        "layout" :                    "regular",   # regular, mlp
        "resolution" :                 16,       
        "num_gaussians_in_mixture" :   1,
        "interpolation" :             "Nearest",   # [Nearest, Linear, Smooth]
        "gaussian_mean_encoding":     "raw",       # [raw, eps-norm, min-max-norm]
        "gaussian_variance_encoding": "softplus",  # [exp, sigmoid, softplus]
        "vmf_sharpness_encoding":     "softplus",  # [exp, relu, sigmoid, softplus]
        "vmf_axis_encoding":          "normalize", # [raw, normalize, spherical, spherical-norm]
        "vmf_amplitude_encoding":     "exp",       # [relu, softplus, exp]
        "accumulate_gaussians" :       False,
    }, 
    # It is also possible to pass optimizer type here
    # but right now I don't see the reason to do that
    "optimizer" : {
        "learning_rate" :  0.001,
        "regularization" : False,
    },
    "epoch" : 1000,
    "spp" : 1
}

config = Config(raw_config)
