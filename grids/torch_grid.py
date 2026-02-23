import torch

from .base import vapl_grid_base


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

        n_levels = config.grid.n_levels if config.grid.n_levels > 1 else self.config.grid.num_neighbours_to_sample + 1

        # gaussian(4) + vmf(8) = 12 input features per level
        feature_dim = self.num_param_per_gaussian + self.num_param_per_vmf
        # shared MLP takes features + level embedding
        level_embed_dim = 4
        input_dim = feature_dim + level_embed_dim
        hidden_dim = 32

        self.level_embedding = torch.nn.Embedding(n_levels, level_embed_dim)

        # Shared MLP: input -> hidden -> hidden -> residual correction
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, feature_dim),
        )

        # Initialize last layer near zero so residual starts as identity
        torch.nn.init.zeros_(self.mlp[-1].weight)
        torch.nn.init.zeros_(self.mlp[-1].bias)

        self.optimizer = torch.optim.Adam(
            list(self.gaussian_grid.parameters()) +
            list(self.vmf_grid.parameters()) +
            list(self.mlp.parameters()) +
            list(self.level_embedding.parameters()),
            lr=self.learning_rate
        )

    def forward(self, input):
        gaussians_list, vmf_list = self.get_vapls(input)

        output_gaussians_list = []
        output_vmf_list = []

        for level, (gauss, vmf) in enumerate(zip(gaussians_list, vmf_list)):
            combined = torch.cat([gauss, vmf], dim=1)

            level_idx = torch.full((combined.shape[0],), level, device=combined.device, dtype=torch.long)
            level_embed = self.level_embedding(level_idx)
            mlp_input = torch.cat([combined, level_embed], dim=1)

            # Residual: grid features + learned correction
            correction = self.mlp(mlp_input)
            refined = combined + correction

            output_gaussians_list.append(refined[:, :self.num_param_per_gaussian])
            output_vmf_list.append(refined[:, self.num_param_per_gaussian:])

        return self.encode(output_gaussians_list, output_vmf_list)
