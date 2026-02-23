class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__(config)
        for key, value in config.items():
            if isinstance(value, dict):
                value = Config(value)
            self[key] = value
        self.__dict__ = self

raw_config = {
    "scene"     : "scenes/country_kitchen/scene.xml", # cornell box, veach_mis
    "mode"      : "local",       # local, wandb, sweep
    "wandb_group"  : "default",
    "sweep_config" : None,       # add to use sweep
    "run_name"     : None,       # for wandb runs
    "grid" : {
        "layout" :                    "drjit",   # regular, mlp, drjit, drjit-mlp
        "resolution" :                 16,
        "n_levels" :                   1,
        "interpolation" :             "Nearest",   # [Nearest, Linear, Smoothstep]
        "gaussian_mean_encoding":     "eps-norm",       # [raw, eps-norm]
        "gaussian_variance_encoding": "sigmoid",  # [exp, sigmoid, softplus]
        "vmf_amplitude_encoding":     "exp",       # [relu, softplus, exp]
        "vmf_axis_encoding":          "raw", # [raw, normalize, spherical, spherical-norm]
        "vmf_sharpness_encoding":     "relu",  # [exp, relu, sigmoid, softplus]
        "accumulate_gaussians" :       False,
        "accumulate_radiance" :        False,       # radiance or parameters of vapls
        "num_neighbours_to_sample":    3,          # when accumulate radiance by neigbours
        "stochastic_interpolation":    True,
        "stochastic_std":              0.1,
    },
    "loss" : "relative_l2",
    # It is also possible to pass optimizer type here
    # but right now I don't see the reason to do that
    "optimizer" : {
        "learning_rate" :  0.001,
        "regularization" : False,
    },
    "epoch" : 501,
    "spp" : 1,
    "depth" : 8
}

config = Config(raw_config)
