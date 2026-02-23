import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import torch
import torch.nn.functional as F
import tinycudann as tcnn
torch.autograd.set_detect_anomaly(True)

import random


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
    "tanh" : torch.tanh,
    "softplus": F.softplus,
    "normalize": F.normalize,
    "spherical": spherical,
    "spherical-norm": spherical_norm,
    "eps-norm" : eps_norm,
    "tanh-norm": lambda x: 0.5 * (torch.tanh(x) + 1),
    "raw": lambda x: x,
}


class vapl_grid_base(torch.nn.Module):
    def __init__(self, config, bb_min, bb_max):
        super().__init__()

        self.config = config
        self.bb_min = torch.tensor(bb_min, device="cuda")
        self.bb_max = torch.tensor(bb_max, device="cuda")

        self.num_param_per_gaussian = 4
        self.num_param_per_vmf = 8

        grid_config = {
            "encoding": {
                "otype": "HashGrid",
                "base_resolution": config.grid.resolution,
                "n_levels": config.grid.n_levels,
                "n_features_per_level": self.num_param_per_gaussian,
                "log2_hashmap_size": 22,
                "interpolation": config.grid.interpolation
            },
        }

        n_input_dims = 3
        self.gaussian_grid = tcnn.Encoding(n_input_dims, grid_config["encoding"])

        grid_config["encoding"]["n_features_per_level"] = self.num_param_per_vmf
        self.vmf_grid = tcnn.Encoding(n_input_dims, grid_config["encoding"])

        self.learning_rate = config.optimizer.learning_rate

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    @classmethod
    def create_vapl_grid(cls, config, bb_min, bb_max):
        if config.grid.layout == "drjit":
            from .drjit_grid import vapl_grid_drjit
            return vapl_grid_drjit(config, bb_min, bb_max)
        elif config.grid.layout == "drjit-mlp":
            from .drjit_mlp_grid import vapl_grid_mlp_drjit
            return vapl_grid_mlp_drjit(config, bb_min, bb_max)
        elif config.grid.layout == "mlp":
            from .torch_grid import vapl_grid_mlp
            return vapl_grid_mlp(config, bb_min, bb_max).cuda()
        else:
            from .torch_grid import vapl_grid
            return vapl_grid(config, bb_min, bb_max).cuda()

    def set_config(self, config):
        self.config = config

    def sample_vpls(self, pos):
        normalized_pos = (pos - self.bb_min) / (self.bb_max - self.bb_min)

        if (self.config.grid.n_levels > 1):
            grid_output = self.gaussian_grid(normalized_pos).to(dtype=torch.float32)
            gaussians_split = grid_output.view(grid_output.shape[0], -1, self.num_param_per_gaussian)
            gaussians_list = [gaussians_split[:, i, :] for i in range(gaussians_split.shape[1])]

            grid_output = self.vmf_grid(normalized_pos).to(dtype=torch.float32)
            vmf_split = grid_output.view(grid_output.shape[0], -1, self.num_param_per_vmf)
            vmf_list = [vmf_split[:, i, :] for i in range(vmf_split.shape[1])]
        else:
            block_size = 1.0 / self.config.grid.resolution
            gaussians_list = [self.gaussian_grid(normalized_pos).to(dtype=torch.float32)]
            vmf_list = [self.vmf_grid(normalized_pos).to(dtype=torch.float32)]
            neighbor_offsets = [
                torch.tensor([dx, dy, dz], device="cuda") * block_size
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for dz in [-1, 0, 1]
                if not (dx == 0 and dy == 0 and dz == 0)
            ]

            random_neighbors = random.sample(neighbor_offsets, self.config.grid.num_neighbours_to_sample)

            for offset in random_neighbors:
                neighbor_pos = normalized_pos + offset

                gaussians = self.gaussian_grid(neighbor_pos).to(dtype=torch.float32)
                vmf = self.vmf_grid(neighbor_pos).to(dtype=torch.float32)

                gaussians_list.append(gaussians)
                vmf_list.append(vmf)

        return gaussians_list, vmf_list

    def sample_vpls_accumulate_params(self, pos):
        block_size = 1.0 / self.config.grid.resolution
        normalized_pos = (pos - self.bb_min) / (self.bb_max - self.bb_min)

        if (self.config.grid.n_levels > 1):
            grid_output = self.gaussian_grid(normalized_pos).to(dtype=torch.float32)
            gaussians_split = grid_output.view(grid_output.shape[0], -1, self.num_param_per_gaussian)
            gaussians_list = [gaussians_split[:, i, :] for i in range(gaussians_split.shape[1])]

            grid_output = self.vmf_grid(normalized_pos).to(dtype=torch.float32)
            vmf_split = grid_output.view(grid_output.shape[0], -1, self.num_param_per_vmf)
            vmf_list = [vmf_split[:, i, :] for i in range(vmf_split.shape[1])]

            total_gaussians = torch.zeros_like(gaussians_list[0])
            total_vmf = torch.zeros_like(vmf_list[0])

            for g in gaussians_list:
                total_gaussians += g

            for v in vmf_list:
                total_vmf += v
        else:
            total_gaussians = self.gaussian_grid(normalized_pos).to(dtype=torch.float32)
            total_vmf = self.vmf_grid(normalized_pos).to(dtype=torch.float32)

            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == 0 and dy == 0 and dz == 0:
                            continue

                        offset = torch.tensor([dx, dy, dz], device="cuda") * block_size
                        neighbor_pos = normalized_pos + offset

                        gaussians = self.gaussian_grid(neighbor_pos).to(dtype=torch.float32)
                        vmf = self.vmf_grid(neighbor_pos).to(dtype=torch.float32)

                        total_gaussians = total_gaussians + gaussians
                        total_vmf = total_vmf + vmf

        return total_gaussians, total_vmf

    def sweep_encoding(self, gaussians, vmf):
        def process_tensors(gaussians, vmf):
            if (self.config.sweep_config.gaussian_mean_encoding == "raw"):
                mean = encoders[self.config.sweep_config.gaussian_mean_encoding](gaussians[:, :3])
            else:
                mean = encoders[self.config.sweep_config.gaussian_mean_encoding](gaussians[:, :3])
                mean = mean * (self.bb_max - self.bb_min) + self.bb_min

            variance = encoders[self.config.sweep_config.gaussian_variance_encoding](gaussians[:, 3]).unsqueeze(1)
            sharpness = encoders[self.config.sweep_config.vmf_sharpness_encoding](vmf[:, 0]).unsqueeze(1)

            if (self.config.sweep_config.vmf_axis_encoding == "spherical" or
                self.config.sweep_config.vmf_axis_encoding == "spherical-norm"):
                axis = encoders[self.config.sweep_config.vmf_axis_encoding](vmf[:, 1:3])
                amplitude = encoders[self.config.sweep_config.vmf_amplitude_encoding](vmf[:, 3:6])
            else:
                axis = encoders[self.config.sweep_config.vmf_axis_encoding](vmf[:, 1:4])
                amplitude = encoders[self.config.sweep_config.vmf_amplitude_encoding](vmf[:, 4:7])

            gaussians = torch.cat([mean, variance], dim=1)
            vmf = torch.cat([sharpness, axis, amplitude], dim=1)

            return gaussians, vmf

        if isinstance(gaussians, list) and isinstance(vmf, list):
            encoded_gaussians = []
            encoded_vmf = []
            for g, v in zip(gaussians, vmf):
                e_gaussians, e_vmf = process_tensors(g, v)
                encoded_gaussians.append(e_gaussians)
                encoded_vmf.append(e_vmf)
            return encoded_gaussians, encoded_vmf
        else:
            return process_tensors(gaussians, vmf)

    def encoding(self, gaussians, vmf):
        def process_tensors(gaussians, vmf):
            if self.config.grid.gaussian_mean_encoding == "raw":
                mean = encoders[self.config.grid.gaussian_mean_encoding](gaussians[:, :3])
            else:
                mean = encoders[self.config.grid.gaussian_mean_encoding](gaussians[:, :3])
                mean = mean * (self.bb_max - self.bb_min) + self.bb_min

            variance = encoders[self.config.grid.gaussian_variance_encoding](gaussians[:, 3]).unsqueeze(1)
            sharpness = encoders[self.config.grid.vmf_sharpness_encoding](vmf[:, 0]).unsqueeze(1)

            if (self.config.grid.vmf_axis_encoding == "spherical" or
                self.config.grid.vmf_axis_encoding == "spherical-norm"):
                axis = encoders[self.config.grid.vmf_axis_encoding](vmf[:, 1:3])
                amplitude = encoders[self.config.grid.vmf_amplitude_encoding](vmf[:, 3:6])
            else:
                axis = encoders[self.config.grid.vmf_axis_encoding](vmf[:, 1:4])
                amplitude = encoders[self.config.grid.vmf_amplitude_encoding](vmf[:, 4:7])

            gaussians = torch.cat([mean, variance], dim=1)
            vmf = torch.cat([sharpness, axis, amplitude], dim=1)

            return gaussians, vmf

        if isinstance(gaussians, list) and isinstance(vmf, list):
            encoded_gaussians = []
            encoded_vmf = []
            for g, v in zip(gaussians, vmf):
                e_gaussians, e_vmf = process_tensors(g, v)
                encoded_gaussians.append(e_gaussians)
                encoded_vmf.append(e_vmf)
            return encoded_gaussians, encoded_vmf
        else:
            return process_tensors(gaussians, vmf)

    def encode(self, gaussians, vmf):
        if self.config.mode == "sweep":
            return self.sweep_encoding(gaussians, vmf)
        else:
            return self.encoding(gaussians, vmf)

    def get_vapls(self, input):
        if isinstance(input, mi.SurfaceInteraction3f):
            pos = input.p.torch().permute(1, 0)

            if self.config.grid.stochastic_interpolation:
                offset = torch.randn_like(pos[:, :2]) * self.config.grid.stochastic_std

                s = input.sh_frame.s.torch().permute(1, 0)
                t = input.sh_frame.t.torch().permute(1, 0)

                offset_world = s * offset[:, :1] + t * offset[:, 1:]
                pos = pos + offset_world
        elif isinstance(input, torch.Tensor):
            pos = input

        if self.config.grid.accumulate_gaussians == True:
            if self.config.grid.accumulate_radiance == True:
                return self.sample_vpls(pos)
            else:
                return self.sample_vpls_accumulate_params(pos)
        else:
            X = (pos - self.bb_min) / (self.bb_max - self.bb_min)
            gaussians : torch.Tensor = self.gaussian_grid(X).to(dtype=torch.float32)
            vmf : torch.Tensor = self.vmf_grid(X).to(dtype=torch.float32)

        return list(gaussians.split(4, dim=1)), list(vmf.split(8, dim=1))

    def get_gaussians_for_debug_render(self):
        with torch.no_grad():
            base_resolution = self.config.grid.resolution
            n_levels = self.config.grid.n_levels
            device = "cuda"

            gaussians_list = []
            vmf_list = []

            for level in range(n_levels):
                resolution = base_resolution * (2 ** level)

                lin = torch.linspace(0, 1, resolution, device=device)
                X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')
                grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)

                world_positions = grid_points * (self.bb_max - self.bb_min) + self.bb_min

                encoded_gaussians, encoded_vmf = self.forward(world_positions)

                gaussians_list.append(encoded_gaussians[level])
                vmf_list.append(encoded_vmf[level])

            return gaussians_list, vmf_list
