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
        "resolution" :                 2,
        "n_levels" :                   4,
        "interpolation" :             "Nearest",   # [Nearest, Linear, Smooth]
        "gaussian_mean_encoding":     "eps-norm",       # [raw, eps-norm]
        "gaussian_variance_encoding": "sigmoid",  # [exp, sigmoid, softplus]
        "vmf_amplitude_encoding":     "exp",       # [relu, softplus, exp]
        "vmf_axis_encoding":          "raw", # [raw, normalize, spherical, spherical-norm]
        "vmf_sharpness_encoding":     "softplus",  # [exp, relu, sigmoid, softplus]
        "accumulate_gaussians" :       True,
        "accumulate_radiance" :        True,       # radiance or parameters of vapls
        "num_neighbours_to_sample":    3,          # when accumulate radiance by neigbours
    },
    # It is also possible to pass optimizer type here
    # but right now I don't see the reason to do that
    "optimizer" : {
        "learning_rate" :  0.001,
        "regularization" : False,
    },
    "epoch" : 201,
    "spp" : 1,
    "depth" : 1
}

config = Config(raw_config)
