import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

import inspect
import math
import torch
import tinycudann as tcnn
torch.autograd.set_detect_anomaly(True)

import matplotlib.pyplot as plt
import numpy as np

class vapl_grid_base(torch.nn.Module):
    def __init__(self, config, bb_min, bb_max):
        super().__init__()

        self.bb_min = bb_min
        self.bb_max = bb_max

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
    def create_vapl_grid(config):
        if config.grid.type == "mlp":
            return vapl_grid_mlp(config)
        else:
            return vapl_grid(config)
        
    def sample_vpls(self, pos):
        bb_min = torch.tensor(self.bb_min, device="cuda")
        bb_max = torch.tensor(self.bb_max, device="cuda")
        block_size = 1.0 / 16.0
        normalized_pos = (pos - bb_min) / (bb_max - bb_min)

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

class vapl_grid(torch.nn.Module):
    def __init__(self, bb_min, bb_max, num_gaussian_in_mixture, num_param_per_gaussian, num_param_per_vmf, sweep_config = None):
        super().__init__()

        self.bb_min = bb_min
        self.bb_max = bb_max
        self.sweep_config = sweep_config

        # tiny-cuda-nn config for hash grid
        config = {
            "encoding": {
                "otype": "HashGrid",
                "base_resolution": 16,
                "n_levels": num_gaussian_in_mixture,
                "n_features_per_level": num_param_per_gaussian,
                "log2_hashmap_size": 22,
                "interpolation": "Nearest"
            },
        }
        n_input_dims = 3
        self.gaussian_grid = tcnn.Encoding(n_input_dims, config["encoding"])

        config["encoding"]["n_features_per_level"] = num_param_per_vmf
        self.vmf_grid = tcnn.Encoding(n_input_dims, config["encoding"])

        learning_rate = 0.001
        if self.sweep_config != None:
            learning_rate = sweep_config.learning_rate

        self.optimizer = torch.optim.Adam(
            list(self.gaussian_grid.parameters()) + list(self.vmf_grid.parameters()),
            lr=learning_rate
        )

        # It is possible to change learning rate during training
        #torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.95)

    def forward(self, input):
        bb_min = torch.tensor(self.bb_min, device="cuda")
        bb_max = torch.tensor(self.bb_max, device="cuda")

        if isinstance(input, mi.SurfaceInteraction3f):
            pos = input.p.torch().permute(1, 0)
        elif isinstance(input, torch.Tensor):
            pos = input

        # TODO: add parameter to switch between accumulated/regular vapls
        #X = (pos - bb_min) / (bb_max - bb_min)
        #gaussians : torch.Tensor = self.gaussian_grid(X).to(dtype=torch.float32)
        #vmf : torch.Tensor = self.vmf_grid(X).to(dtype=torch.float32)

        gaussians, vmf = self.sample_vpls(pos)

        if self.sweep_config != None:
            if (self.sweep_config.gaussian_mean_encoding == "raw"):
                mean = encoders[self.sweep_config.gaussian_mean_encoding](gaussians[:, :3])
            else:
                mean = encoders[self.sweep_config.gaussian_mean_encoding](gaussians[:, :3]) * (bb_max - bb_min) + bb_min

            variance = encoders[self.sweep_config.gaussian_variance_encoding](gaussians[:, 3]).unsqueeze(1)
            sharpness = encoders[self.sweep_config.vmf_sharpness_encoding](vmf[:, 0]).unsqueeze(1)

            if (self.sweep_config.vmf_axis_encoding == "sperical" or self.sweep_config.vmf_axis_encoding == "spherical-norm"):
                axis = encoders[self.sweep_config.vmf_axis_encoding](vmf[:, 1:3])
                amplitude = encoders[self.sweep_config.vmf_amplitude_encoding](vmf[:, 3:6])
            else:
                axis = encoders[self.sweep_config.vmf_axis_encoding](vmf[:, 1:4])
                amplitude = encoders[self.sweep_config.vmf_amplitude_encoding](vmf[:, 4:7])
        else:
            mean = (gaussians[:, :3] - gaussians[:, :3].min()) / (gaussians[:, :3].max() - gaussians[:, :3].min())
            mean = mean * (bb_max - bb_min) + bb_min
            variance = torch.exp(gaussians[:, 3]).unsqueeze(1)
            sharpness = torch.sigmoid(vmf[:, 0]).unsqueeze(1)

            #theta = torch.sigmoid(vmf[:, 1])
            #phi   = torch.sigmoid(vmf[:, 2])
            #axis = torch.stack([
            #     torch.sin(theta) * torch.cos(phi),
            #     torch.sin(theta) * torch.sin(phi),
            #     torch.cos(theta)]).permute(1, 0)
            #axis = torch.nn.functional.normalize(axis, p=2, dim=1, eps=1e-6)
            axis = vmf[:, 1:4]
            amplitude = torch.exp(vmf[:, 4:7])

        gaussians = torch.cat([mean, variance], dim = 1)
        vmf = torch.cat([sharpness, axis, amplitude], dim = 1)

        return gaussians, vmf

class vapl_grid_mlp(vapl_grid_base):
    def __init__(self, bb_min, bb_max, num_gaussian_in_mixture, num_param_per_gaussian, num_param_per_vmf):
        super().__init__()

        self.bb_min = bb_min
        self.bb_max = bb_max

        config = {
            "encoding": {
                "otype": "HashGrid",
                "base_resolution": 16,
                "n_levels": num_gaussian_in_mixture,
                "n_features_per_level": num_param_per_gaussian,
                "log2_hashmap_size": 22,
                "interpolation": "Nearest"
            },
        }
        n_input_dims = 3
        self.gaussian_grid = tcnn.Encoding(n_input_dims, config["encoding"])

        config["encoding"]["n_features_per_level"] = num_param_per_vmf
        self.vmf_grid = tcnn.Encoding(n_input_dims, config["encoding"])

        layers = []
        input_dim = num_param_per_gaussian + num_param_per_vmf
        hidden_dim = 32
        output_dim = 11

        for _ in range(3):
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.LeakyReLU(negative_slope=0.2))
            input_dim = hidden_dim

        layers.append(torch.nn.Linear(hidden_dim, output_dim))
        self.fc = torch.nn.Sequential(*layers)

        self.optimizer = torch.optim.Adam(
            list(self.gaussian_grid.parameters()) +
            list(self.vmf_grid.parameters()) +
            list(self.fc.parameters()),
            lr=0.001
        )

    def forward(self, input):
        bb_min = torch.tensor(self.bb_min, device="cuda")
        bb_max = torch.tensor(self.bb_max, device="cuda")

        if isinstance(input, mi.SurfaceInteraction3f):
            pos = input.p.torch().permute(1, 0)
        elif isinstance(input, torch.Tensor):
            pos = input

        X = (pos - bb_min) / (bb_max - bb_min)

        gaussians = self.gaussian_grid(X).to(dtype=torch.float32)
        vmf = self.vmf_grid(X).to(dtype=torch.float32)
        grid_output = torch.cat([gaussians, vmf], dim=1)

        outputs = self.fc(grid_output)

        eps = 1
        mean = torch.sigmoid(outputs[:, :3]) * (bb_max - bb_min) + bb_min
        variance = torch.nn.functional.relu(outputs[:, 3:4])
        sharpness = torch.nn.functional.relu(outputs[:, 4:5])

        axis = torch.nn.functional.normalize(outputs[:, 5:8], p=2, dim=1, eps=1e-6)
        amplitude = torch.sigmoid(outputs[:, 8:11])

        gaussians_out = torch.cat([mean, variance], dim=1)
        vmf_out = torch.cat([sharpness, axis, amplitude], dim=1)

        return gaussians_out, vmf_out
