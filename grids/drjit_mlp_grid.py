import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from drjit.auto.ad import Float16, TensorXf16

import numpy as np

from .drjit_grid import vapl_grid_drjit


class vapl_grid_mlp_drjit(vapl_grid_drjit):

    def __init__(self, config, bb_min, bb_max):
        super().__init__(config, bb_min, bb_max)

        n_levels = config.grid.n_levels if config.grid.n_levels > 1 else config.grid.num_neighbours_to_sample + 1

        feature_dim = self.num_param_per_gaussian + self.num_param_per_vmf  # 12
        level_embed_dim = 4
        input_dim = feature_dim + level_embed_dim
        hidden_dim = 32

        self.n_levels_mlp = n_levels
        self.level_embed_dim = level_embed_dim
        self.feature_dim = feature_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Build MLP using DrJIT nn module (tensor-core accelerated matvec)
        mlp_structure = dr.nn.Sequential(
            dr.nn.Linear(input_dim, hidden_dim),
            dr.nn.ReLU(),
            dr.nn.Linear(hidden_dim, hidden_dim),
            dr.nn.ReLU(),
            dr.nn.Linear(hidden_dim, feature_dim),
        )
        mlp_alloc = mlp_structure.alloc(TensorXf16, input_dim)

        # Scale last layer near zero for residual identity initialization
        last_layer = mlp_alloc.layers[4]
        last_layer.weights = TensorXf16(last_layer.weights * 0.01)

        # Pack into optimal GPU layout for evaluation
        self.mlp_coeffs, self.mlp = dr.nn.pack(mlp_alloc, layout='training')

        # Store packed coefficients in optimizer (Float32 for precision)
        self.opt['mlp'] = mi.Float(self.mlp_coeffs.array)

        # Trainable level embeddings
        self.opt['level_embed'] = mi.Float(dr.zeros(mi.Float, n_levels * level_embed_dim))

    def __call__(self, si_or_pos):
        if isinstance(si_or_pos, mi.SurfaceInteraction3f):
            pos = si_or_pos.p
            if self.config.grid.stochastic_interpolation:
                n = dr.width(pos)
                offset_s = mi.Float(np.random.normal(size=n).astype(np.float32)) * self.config.grid.stochastic_std
                offset_t = mi.Float(np.random.normal(size=n).astype(np.float32)) * self.config.grid.stochastic_std
                s = si_or_pos.sh_frame.s
                t = si_or_pos.sh_frame.t
                pos = pos + s * offset_s + t * offset_t
        else:
            pos = si_or_pos

        normalized = self.normalize_pos(pos)
        n = dr.width(normalized)

        g_levels, v_levels = self.query_grids(normalized)

        # Evaluate all raw grid outputs upfront
        for cols in g_levels + v_levels:
            dr.eval(*cols)

        # Sync MLP weights from optimizer (creates AD connection for backward)
        self.mlp_coeffs.array[:] = Float16(self.opt['mlp'])

        level_embed = self.opt['level_embed']

        # --- Batch all levels into a single MLP call ---
        # With n_levels separate MLP calls, dr.backward() must compile
        # n_levels separate backward MLP kernels which gets too slow at n_levels>=4.
        # Batching into one call means one forward + one backward MLP kernel.
        n_levels = len(g_levels)
        total_n = n * n_levels

        # Build batched input: concatenate features from all levels
        batched_features = [dr.zeros(mi.Float, total_n) for _ in range(self.input_dim)]

        all_combined = []  # keep originals for residual connection
        for level, (g_cols, v_cols) in enumerate(zip(g_levels, v_levels)):
            combined = list(g_cols) + list(v_cols)
            all_combined.append(combined)

            level_emb = []
            for d in range(self.level_embed_dim):
                flat_idx = level * self.level_embed_dim + d
                val = dr.gather(mi.Float, level_embed, dr.full(mi.UInt32, flat_idx, n))
                level_emb.append(val)

            mlp_input = combined + level_emb
            idx = dr.arange(mi.UInt32, n) + level * n
            for i, feat in enumerate(mlp_input):
                dr.scatter(batched_features[i], feat, idx)

        # Materialize the batched buffer before the MLP
        dr.eval(*batched_features)

        # Single MLP forward for all levels at once
        cv_in = dr.nn.CoopVec(*batched_features)
        cv_in = dr.nn.cast(cv_in, Float16)
        cv_out = self.mlp(cv_in)
        corrections_all = [mi.Float(f) for f in list(cv_out)]
        dr.eval(*corrections_all)

        # Split corrections back per level and apply residual + encoding
        gaussians_list = []
        vmfs_list = []

        for level in range(n_levels):
            idx = dr.arange(mi.UInt32, n) + level * n
            correction = [dr.gather(mi.Float, c, idx) for c in corrections_all]

            combined = all_combined[level]
            refined = [c + corr for c, corr in zip(combined, correction)]

            g_refined = refined[:self.num_param_per_gaussian]
            v_refined = refined[self.num_param_per_gaussian:]

            mean, variance = self.encoding_gaussian(g_refined, normalized)
            sharpness, axis, amplitude = self.encoding_vmf(v_refined)

            dr.eval(mean, variance, sharpness, axis, amplitude)

            gaussians_list.append((mean, variance))
            vmfs_list.append((sharpness, axis, amplitude))

        return gaussians_list, vmfs_list
