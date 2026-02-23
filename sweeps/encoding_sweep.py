encoding_sweep_config = {
    "method": "bayes",
    "program": "sweeps/train.py",
    "metric": {"name": "loss", "goal": "minimize"},
    "parameters": {
        "gaussian_mean_encoding":     {"values": ["raw", "sigmoid", "eps-norm", "tanh-norm"]},
        "gaussian_variance_encoding": {"values": ["exp", "sigmoid", "softplus"]},
        "vmf_sharpness_encoding":     {"values": ["exp", "relu", "sigmoid", "softplus"]},
        "vmf_axis_encoding":          {"values": ["raw", "normalize", "spherical", "spherical-norm"]},
        "vmf_amplitude_encoding":     {"values": ["relu", "softplus", "exp", "sigmoid"]},
        "epoch":                      {"values": [200]},
    },
}
