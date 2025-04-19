import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import torch
import torch.nn.functional as F
import tinycudann as tcnn
torch.autograd.set_detect_anomaly(True)

import numpy as np
import matplotlib.pyplot as plt

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
    "tanh" : torch.tanh,
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
        self.bb_min = torch.tensor(bb_min, device="cuda")
        self.bb_max = torch.tensor(bb_max, device="cuda")
       
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
        self.vmf_grid = tcnn.Encoding(n_input_dims, grid_config["encoding"])

        self.learning_rate = config.optimizer.learning_rate

    @classmethod
    def create_vapl_grid(cls, config , bb_min, bb_max):
        if config.grid.layout == "mlp":
            return vapl_grid_mlp(config, bb_min, bb_max).cuda()
        else:
            return vapl_grid(config, bb_min, bb_max).cuda()
        
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

                    total_gaussians = total_gaussians + gaussians
                    total_vmf = total_vmf + vmf

        return total_gaussians, total_vmf
    
    def sweep_encoding(self, gaussians, vmf):
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
                
        gaussians = torch.cat([mean, variance], dim = 1)
        vmf = torch.cat([sharpness, axis, amplitude], dim = 1)

        return gaussians, vmf
    
    def encoding(self, gaussians, vmf):
        if (self.config.grid.gaussian_mean_encoding == "raw"):
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

        gaussians = torch.cat([mean, variance], dim = 1)
        vmf = torch.cat([sharpness, axis, amplitude], dim = 1)

        return gaussians, vmf

    def encode(self, gaussians, vmf):
        if self.config.mode == "sweep":
            return self.sweep_encoding(gaussians, vmf)
        else:
            return self.encoding(gaussians, vmf)

    def get_vapls(self, input):
        if isinstance(input, mi.SurfaceInteraction3f):
            pos = input.p.torch().permute(1, 0)
        elif isinstance(input, torch.Tensor):
            pos = input

        if self.config.grid.accumulate_gaussians == True:
            return self.sample_vpls(pos)
        else:
            X = (pos - self.bb_min) / (self.bb_max - self.bb_min)
            gaussians : torch.Tensor = self.gaussian_grid(X).to(dtype=torch.float32)
            vmf : torch.Tensor = self.vmf_grid(X).to(dtype=torch.float32)

        return gaussians, vmf
    
    def get_gaussians_for_debug_render(self):
        with torch.no_grad():
            resolution = int(self.config.grid.resolution / 8) # FIXME hack hack
            device = "cuda"

            lin = torch.linspace(0, 1, resolution, device=device)
            X, Y, Z = torch.meshgrid(lin, lin, lin, indexing='ij')

            grid_points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
            world_positions : torch.Tensor = grid_points * (self.bb_max - self.bb_min) + self.bb_min
            
            gaussians : torch.Tensor = self.gaussian_grid(world_positions).to(dtype=torch.float32)
            vmf : torch.Tensor = self.vmf_grid(world_positions).to(dtype=torch.float32)
            return self.encode(gaussians, vmf)
        
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
        vmf = outputs[:, 4:11]

        return self.encode(gaussians, vmf)
    

# helper functions for debug vapl visualization
def world_to_ndc(scene, batch):
    """Transforms 3D world coordinates into normalized device coordinates (NDC) using the perspective transformation matrix.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.

    Returns:
        mi.Point3f: Array of 3D points in NDC.
    """
    sensor = mi.traverse(scene.sensors()[0])
    fov = sensor['x_fov']
    near = sensor['near_clip']
    far = sensor['far_clip']

    trafo = mi.Transform4f().perspective(fov, near, far)
    pts = trafo @ sensor['to_world'].inverse() @ mi.Point3f(np.array(batch.T))
    return pts

def ndc_to_pixel(pts, h, w):
    """Converts points in NDC to pixel coordinates.

    Args:
        pts (mi.Point2f): Points in NDC.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given points.
    """
    hh, hw = h/2, w/2
    return mi.Point2f(dr.fma(pts.x, -hw, hw), dr.fma(pts.y, -hw, hh))  # not typo

def draw_multi_segments(starts, ends, color):
    """Draws multiple line segments on a plot.

    Args:
        starts (mi.Point2f): Starting points of the line segments.
        ends (mi.Point2f): Ending points of the line segments.
        color (str): Color of the line segments.
    """
    a = np.c_[starts.x, starts.y]
    b = np.c_[ends.x, ends.y]
    plt.plot(*np.c_[a, b, a*np.nan].reshape(-1, 2).T, color)

def pix_coord(scene, batch, h, w):
    """Calculates the pixel coordinates of the given 3D world coordinates.

    Args:
        scene (mi.Scene): Mitsuba 3 scene containing the camera information.
        batch (array_like): Array of 3D world coordinates.
        h (float): Height of the image in pixels.
        w (float): Width of the image in pixels.

    Returns:
        mi.Point2f: Pixel coordinates of the given 3D world coordinates.
    """
    return ndc_to_pixel(world_to_ndc(scene, batch), h, w)
