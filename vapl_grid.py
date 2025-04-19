import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import torch
import torch.nn.functional as F
import tinycudann as tcnn
torch.autograd.set_detect_anomaly(True)

def minmaxnorm(x):
    return (x - x.min()) / (x.max() - x.min())

def eps_norm(x):
    eps = 1e-2
    return x / eps * 0.5 - 0.5

def spherical(x):
    theta = torch.sigmoid(x[:, 0])
    phi   = torch.sigmoid(x[:, 1])
    axis = torch.stack([
             torch.sin(theta) * torch.cos(phi),
             torch.sin(theta) * torch.sin(phi),
             torch.cos(theta)]).permute(1, 0)
    return axis

def spherical_norm(x):
    return F.normalize(spherical(x))

encoders = {
    "log": torch.log,
    "exp": torch.exp,
    "relu": torch.relu,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "softplus": F.softplus,
    "normalize": F.normalize,
    "spherical": spherical,
    "spherical-norm": spherical_norm,
    "eps-norm" : eps_norm,
    "min-max-norm": minmaxnorm,
    "raw": lambda x: x,
}

class vapl_grid_base(torch.nn.Module):
    def __init__(self, config, bb_min, bb_max):
        super().__init__()

        self.config = config
        self.bb_min = torch.tensor(self.bb_min, device="cuda")
        self.bb_max = torch.tensor(self.bb_max, device="cuda")
       
        self.num_param_per_gaussian = 4
        self.num_param_per_vmf = 8

        grid_config = {
            "encoding": {
                "otype": "HashGrid",
                "base_resolution": config.grid.resolution,
                "n_levels": config.grid.num_gaussians_in_mixture,
                "n_features_per_level": self.num_param_per_gaussian,
                "log2_hashmap_size": 22,
                "interpolation": config.grid.interpolation
            },
        }

        n_input_dims = 3
        self.gaussian_grid = tcnn.Encoding(n_input_dims, grid_config["encoding"])

        grid_config["encoding"]["n_features_per_level"] = self.num_param_per_vmf
        self.vmf_grid = tcnn.Encoding(n_input_dims, config["encoding"])

        self.learning_rate = config.optimizer.learning_rate
        if (config.mode == "sweep"):
            self.learning_rate = config.sweep_config.learning_rate

    @classmethod
    def create_vapl_grid(config, bb_min, bb_max):
        if config.grid.type == "mlp":
            return vapl_grid_mlp(config, bb_min, bb_max)
        else:
            return vapl_grid(config, bb_min, bb_max)
        
    def sample_vpls(self, pos):
        block_size = 1.0 / self.config.grid.resolution
        normalized_pos = (pos - self.bb_min) / (self.bb_max - self.bb_min)

        total_gaussians = self.gaussian_grid(normalized_pos).to(dtype=torch.float32)
        total_vmf = self.vmf_grid(normalized_pos).to(dtype=torch.float32)

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx==0 and dy==0 and dz==0:
                        continue

                    offset = torch.tensor([dx, dy, dz], device="cuda") * block_size
                    neighbor_pos = normalized_pos + offset

                    gaussians = self.gaussian_grid(neighbor_pos).to(dtype=torch.float32)
                    vmf = self.vmf_grid(neighbor_pos).to(dtype=torch.float32)

                    total_gaussians += gaussians
                    total_vmf += vmf

        return total_gaussians, total_vmf
    
    def sweep_encoding(self, gaussians, vmf):
        if (self.config.sweep_config.gaussian_mean_encoding == "raw"):
            mean = encoders[self.config.sweep_config.gaussian_mean_encoding](gaussians[:, :3])
        else:
            mean = encoders[self.config.sweep_config.gaussian_mean_encoding](gaussians[:, :3])
            mean = mean * (self.bb_max - self.bb_min) + self.bb_min
            variance = encoders[self.config.sweep_config.gaussian_variance_encoding](gaussians[:, 3]).unsqueeze(1)
            sharpness = encoders[self.config.sweep_config.vmf_sharpness_encoding](vmf[:, 0]).unsqueeze(1)

            if (self.config.sweep_config.vmf_axis_encoding == "sperical" or
                 self.config.sweep_config.vmf_axis_encoding == "spherical-norm"):
                axis = encoders[self.config.sweep_config.vmf_axis_encoding](vmf[:, 1:3])
                amplitude = encoders[self.config.sweep_config.vmf_amplitude_encoding](vmf[:, 3:6])
            else:
                axis = encoders[self.config.sweep_config.vmf_axis_encoding](vmf[:, 1:4])
                amplitude = encoders[self.config.sweep_config.vmf_amplitude_encoding](vmf[:, 4:7])
                
        gaussians = torch.cat([mean, variance], dim = 1)
        vmf = torch.cat([sharpness, axis, amplitude], dim = 1)

        return gaussians, vmf
    
    def encoding(self, gaussians, vmf):
        if (self.config.gaussian_mean_encoding == "raw"):
            mean = encoders[self.config.gaussian_mean_encoding](gaussians[:, :3])
        else:
            mean = encoders[self.config.gaussian_mean_encoding](gaussians[:, :3])
            mean = mean * (self.bb_max - self.bb_min) + self.bb_min
            variance = encoders[self.config.gaussian_variance_encoding](gaussians[:, 3]).unsqueeze(1)
            sharpness = encoders[self.config.vmf_sharpness_encoding](vmf[:, 0]).unsqueeze(1)

            if (self.config.vmf_axis_encoding == "sperical" or
                 self.config.vmf_axis_encoding == "spherical-norm"):
                axis = encoders[self.config.vmf_axis_encoding](vmf[:, 1:3])
                amplitude = encoders[self.config.vmf_amplitude_encoding](vmf[:, 3:6])
            else:
                axis = encoders[self.config.vmf_axis_encoding](vmf[:, 1:4])
                amplitude = encoders[self.config.vmf_amplitude_encoding](vmf[:, 4:7])
                
        gaussians = torch.cat([mean, variance], dim = 1)
        vmf = torch.cat([sharpness, axis, amplitude], dim = 1)

        return gaussians, vmf

    def encode(self, gaussians, vmf):
        if self.config.mode == "sweep":
            gaussians, vmf = self.sweep_encoding(gaussians, vmf)
        else:
            gaussians, vmf = self.encoding(gaussians, vmf)

        return gaussians, vmf

    def get_vapls(self, input):
        if isinstance(input, mi.SurfaceInteraction3f):
            pos = input.p.torch().permute(1, 0)
        elif isinstance(input, torch.Tensor):
            pos = input

        if self.config.grid.accumulate_gaussians == True:
            gaussians, vmf = self.sample_vpls(pos)
        else:
            X = (pos - self.bb_min) / (self.bb_max - self.bb_min)
            gaussians : torch.Tensor = self.gaussian_grid(X).to(dtype=torch.float32)
            vmf : torch.Tensor = self.vmf_grid(X).to(dtype=torch.float32)

        return gaussians, vmf
        
class vapl_grid(vapl_grid_base):
    def __init__(self, config, bb_min, bb_max):
        super().__init__(config, bb_min, bb_max)

        self.optimizer = torch.optim.Adam(
            list(self.gaussian_grid.parameters()) + list(self.vmf_grid.parameters()),
            lr=self.learning_rate
        )

        # It is possible to change learning rate during training
        #torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def forward(self, input):
        gaussians, vmf = self.get_vapls(input)
        return self.encode(gaussians, vmf)

class vapl_grid_mlp(vapl_grid_base):
    def __init__(self, config, bb_min, bb_max):
        super().__init__(config, bb_min, bb_max)

        layers = []
        input_dim =  12 #num_param_per_gaussian + num_param_per_vmf
        hidden_dim = 32
        output_dim = 11

        for _ in range(3):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2)) # TODO: set it up properly
            input_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.fc = torch.nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(
            list(self.gaussian_grid.parameters()) +
            list(self.vmf_grid.parameters()) +
            list(self.fc.parameters()),
            lr=self.learning_rate
        )

    def forward(self, input):
        gaussians, vmf = self.get_vapls(input)
        grid_output = torch.cat([gaussians, vmf], dim=1)
        outputs = self.fc(grid_output)

        gaussians = outputs[:, :4]
        vmf = outputs[4:11]
        
        return self.encode(gaussians, vmf)