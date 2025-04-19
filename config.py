class Config(dict):
    def __init__(self, config):
        super(Config, self).__init__(config)
        self.__dict__ = self 

raw_config = {
    "scene"     : "cornell box", # cornell box, veach mis
    "mode"      : "local",       # local, wandb, sweep
    "sweep_config" : None,       # add to use sweep
    "grid" : {
        "type" :                      "regular",   # regular, mlp
        "resolution" :                 16,       
        "num_gaussians_in_mixture" :   1,
        "interpolation" :             "Nearest",   # Nearest, Linear, Smooth
        "gaussian_mean_encoding":     "raw",       # raw, eps-norm, "min-max-norm"
        "gaussian_variance_encoding": "exp",       # exp, sigmoid, softplus
        "vmf_sharpness_encoding":     "exp",       # exp, relu, sigmoid, softplus
        "vmf_axis_encoding":          "normalize", # raw, normalize, spherical, spherical-norm
        "vmf_amplitude_encoding":     "relu",      # relu, softplus, exp
        "accumulate_gaussians" :       True,
    }, 
    "optimizer" : {
        "learning_rate" :  0.001,
        "regularization" : False,
    },
}

config = Config(raw_config)
