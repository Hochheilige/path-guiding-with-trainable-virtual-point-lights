import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from drjit.auto.ad import Float16, TensorXf16
from drjit.hashgrid import HashGridEncoding

import math
import random
import numpy as np

from vapl_utils_drjit import dr_sigmoid, dr_softplus


class VaplHashGridEncoding(HashGridEncoding):
    """HashGridEncoding with configurable interpolation mode."""

    def __init__(self, *args, interpolation="Linear", **kwargs):
        super().__init__(*args, **kwargs)
        self._interpolation = interpolation

    def __call__(self, p, active=True):
        if self._interpolation != "Nearest":
            return super().__call__(p, active)

        # Nearest: single vertex lookup per level (no interpolation)
        _, PositionFloatXf = self._position_types(p)
        p = PositionFloatXf(p)

        out_values = [self.StorageFloat(0.0)] * (
            self.n_features_per_level * self.n_levels
        )

        for level_i in range(self.n_levels):
            scale = self._level_scale(level_i)
            p_offset = 0.5 if not self.align_corners or self.torchngp_compat else 0.0
            pos = dr.fma(p, scale, p_offset)

            # Round to nearest vertex instead of floor + interpolate
            pos_nearest = self.ArrayXu(dr.round(pos))

            index = self.indexing_function(pos_nearest, level_i)
            self._acc_features(level_i, 1.0, index, out_values, active)

        return self.StorageFloatXf(*out_values) & active

def spherical(col0, col1):
    theta = dr_sigmoid(col0)
    phi = dr_sigmoid(col1)
    x = dr.sin(theta) * dr.cos(phi)
    y = dr.sin(theta) * dr.sin(phi)
    z = dr.cos(theta)
    return mi.Vector3f(x, y, z)

def spherical_norm(col0, col1):
    return dr.normalize(spherical(col0, col1))


class vapl_grid_drjit:
    _is_drjit = True

    def __init__(self, config, bb_min, bb_max):
        self.config = config
        self.bb_min = mi.Point3f(bb_min)
        self.bb_max = mi.Point3f(bb_max)
        self.current_epoch = 0

        self.num_param_per_gaussian = 4
        self.num_param_per_vmf = 8
        self.n_levels = config.grid.n_levels
        base_res = config.grid.resolution
        hashmap_size = 2**19

        interp = getattr(config.grid, 'interpolation', 'Linear')

        self.gaussian_enc = VaplHashGridEncoding(
            Float16, 3,
            n_levels=self.n_levels,
            n_features_per_level=self.num_param_per_gaussian,
            hashmap_size=hashmap_size,
            base_resolution=base_res,
            per_level_scale=2.0,
            interpolation=interp,
        )

        self.vmf_enc = VaplHashGridEncoding(
            Float16, 3,
            n_levels=self.n_levels,
            n_features_per_level=self.num_param_per_vmf,
            hashmap_size=hashmap_size,
            base_resolution=base_res,
            per_level_scale=2.0,
            interpolation=interp,
        )

        lr = config.optimizer.learning_rate
        self.opt = dr.opt.Adam(lr=lr)
        self.opt['gaussian'] = mi.Float(self.gaussian_enc.params)
        self.opt['vmf'] = mi.Float(self.vmf_enc.params)
        self.scaler = dr.opt.GradScaler()

    @property
    def optimizer(self):
        return self

    def zero_grad(self):
        pass

    def step(self):
        self.scaler.step(self.opt)

    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def set_config(self, config):
        self.config = config

    def normalize_pos(self, pos):
        extent = self.bb_max - self.bb_min
        return (pos - self.bb_min) / extent

    def query_grids(self, normalized_pos):
        self.gaussian_enc.params[:] = Float16(self.opt['gaussian'])
        self.vmf_enc.params[:] = Float16(self.opt['vmf'])

        cv_in = dr.nn.CoopVec(normalized_pos.x, normalized_pos.y, normalized_pos.z)

        g_out = self.gaussian_enc(cv_in)
        v_out = self.vmf_enc(cv_in)

        g_flat = [mi.Float(f) for f in list(g_out)]
        v_flat = [mi.Float(f) for f in list(v_out)]

        npg = self.num_param_per_gaussian
        npv = self.num_param_per_vmf

        g_levels = [g_flat[l * npg : (l + 1) * npg] for l in range(self.n_levels)]
        v_levels = [v_flat[l * npv : (l + 1) * npv] for l in range(self.n_levels)]

        return g_levels, v_levels

    def encoding_gaussian(self, raw_cols):
        cfg = self.config.grid
        extent = self.bb_max - self.bb_min

        mean_enc = cfg.gaussian_mean_encoding
        if mean_enc == "raw":
            mean = mi.Point3f(raw_cols[0], raw_cols[1], raw_cols[2])
        elif mean_enc == "sigmoid":
            mean = mi.Point3f(dr_sigmoid(raw_cols[0]), dr_sigmoid(raw_cols[1]), dr_sigmoid(raw_cols[2]))
            mean = mean * extent + self.bb_min
        elif mean_enc == "eps-norm":
            eps = 1e-2
            mx = raw_cols[0] / eps * 0.5 - 0.5
            my = raw_cols[1] / eps * 0.5 - 0.5
            mz = raw_cols[2] / eps * 0.5 - 0.5
            mean = mi.Point3f(mx, my, mz)
            mean = mean * extent + self.bb_min
        else:
            mean = mi.Point3f(raw_cols[0], raw_cols[1], raw_cols[2])
            mean = mean * extent + self.bb_min

        var_enc = cfg.gaussian_variance_encoding
        if var_enc == "exp":
            variance = dr.exp(raw_cols[3])
        elif var_enc == "sigmoid":
            variance = dr_sigmoid(raw_cols[3])
        elif var_enc == "softplus":
            variance = dr_softplus(raw_cols[3])
        else:
            variance = raw_cols[3]

        return mean, variance

    def encoding_vmf(self, raw_cols):
        cfg = self.config.grid

        s_enc = cfg.vmf_sharpness_encoding
        if s_enc == "exp":
            sharpness = dr.exp(raw_cols[0])
        elif s_enc == "sigmoid":
            sharpness = dr_sigmoid(raw_cols[0])
        elif s_enc == "softplus":
            sharpness = dr_softplus(raw_cols[0])
        elif s_enc == "relu":
            sharpness = dr.maximum(raw_cols[0], 0.0)
        else:
            sharpness = raw_cols[0]

        a_enc = cfg.vmf_axis_encoding
        if a_enc == "spherical":
            axis = spherical(raw_cols[1], raw_cols[2])
            amp_start = 3
        elif a_enc == "spherical-norm":
            axis = spherical_norm(raw_cols[1], raw_cols[2])
            amp_start = 3
        elif a_enc == "normalize":
            axis = dr.normalize(mi.Vector3f(raw_cols[1], raw_cols[2], raw_cols[3]))
            amp_start = 4
        else:  # "raw"
            axis = mi.Vector3f(raw_cols[1], raw_cols[2], raw_cols[3])
            amp_start = 4

        amp_enc = cfg.vmf_amplitude_encoding
        r, g, b = raw_cols[amp_start], raw_cols[amp_start + 1], raw_cols[amp_start + 2]
        if amp_enc == "exp":
            amplitude = mi.Color3f(dr.exp(r), dr.exp(g), dr.exp(b))
        elif amp_enc == "relu":
            amplitude = mi.Color3f(dr.maximum(r, 0.0), dr.maximum(g, 0.0), dr.maximum(b, 0.0))
        elif amp_enc == "softplus":
            amplitude = mi.Color3f(dr_softplus(r), dr_softplus(g), dr_softplus(b))
        elif amp_enc == "sigmoid":
            amplitude = mi.Color3f(dr_sigmoid(r), dr_sigmoid(g), dr_sigmoid(b))
        else:
            amplitude = mi.Color3f(r, g, b)

        return sharpness, axis, amplitude

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

        if self.config.grid.accumulate_gaussians and self.config.grid.n_levels == 1:
            return self.query_with_neighbours(normalized)

        g_levels, v_levels = self.query_grids(normalized)

        gaussians_list = []
        vmfs_list = []
        for g_cols, v_cols in zip(g_levels, v_levels):
            mean, variance = self.encoding_gaussian(g_cols)
            sharpness, axis, amplitude = self.encoding_vmf(v_cols)
            gaussians_list.append((mean, variance))
            vmfs_list.append((sharpness, axis, amplitude))

        return gaussians_list, vmfs_list

    def query_with_neighbours(self, normalized_pos):
        block_size = 1.0 / self.config.grid.resolution

        g_levels, v_levels = self.query_grids(normalized_pos)
        g_cols = g_levels[0]
        v_cols = v_levels[0]

        mean, variance = self.encoding_gaussian(g_cols)
        sharpness, axis, amplitude = self.encoding_vmf(v_cols)
        gaussians_list = [(mean, variance)]
        vmfs_list = [(sharpness, axis, amplitude)]

        offsets = [
            mi.Vector3f(dx, dy, dz) * block_size
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        num_to_sample = self.config.grid.num_neighbours_to_sample
        chosen = random.sample(offsets, min(num_to_sample, len(offsets)))

        for offset in chosen:
            npos = normalized_pos + offset
            gl, vl = self.query_grids(npos)
            m, var = self.encoding_gaussian(gl[0])
            sh, ax, amp = self.encoding_vmf(vl[0])
            gaussians_list.append((m, var))
            vmfs_list.append((sh, ax, amp))

        return gaussians_list, vmfs_list

    def get_gaussians_for_debug_render(self):
        base_resolution = self.config.grid.resolution
        n_levels = self.config.grid.n_levels

        gaussians_list = []
        vmf_list = []

        for level in range(n_levels):
            resolution = base_resolution * (2 ** level)

            lin = np.linspace(0, 1, resolution, dtype=np.float32)
            xx, yy, zz = np.meshgrid(lin, lin, lin, indexing='ij')
            grid_pts = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)

            extent = self.bb_max - self.bb_min
            world_pos = mi.Point3f(
                mi.Float(grid_pts[:, 0]) * extent.x + self.bb_min.x,
                mi.Float(grid_pts[:, 1]) * extent.y + self.bb_min.y,
                mi.Float(grid_pts[:, 2]) * extent.z + self.bb_min.z,
            )

            g_all, v_all = self(world_pos)
            idx = min(level, len(g_all) - 1)
            mean, variance = g_all[idx]
            sharpness, axis, amplitude = v_all[idx]

            mean_np = np.column_stack([
                np.array(mean.x), np.array(mean.y), np.array(mean.z),
                np.array(variance)
            ])
            vmf_np = np.column_stack([
                np.array(sharpness),
                np.array(axis.x), np.array(axis.y), np.array(axis.z),
                np.array(amplitude.x), np.array(amplitude.y), np.array(amplitude.z),
            ])

            import torch
            gaussians_list.append(torch.from_numpy(mean_np).cuda())
            vmf_list.append(torch.from_numpy(vmf_np).cuda())

        return gaussians_list, vmf_list


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

        # Sync MLP weights from optimizer (creates AD connection for backward)
        self.mlp_coeffs.array[:] = Float16(self.opt['mlp'])

        level_embed = self.opt['level_embed']

        gaussians_list = []
        vmfs_list = []

        for level, (g_cols, v_cols) in enumerate(zip(g_levels, v_levels)):
            combined = list(g_cols) + list(v_cols)

            # Level embedding (broadcast scalar to width n)
            level_emb = []
            for d in range(self.level_embed_dim):
                flat_idx = level * self.level_embed_dim + d
                val = dr.gather(mi.Float, level_embed, dr.full(mi.UInt32, flat_idx, n))
                level_emb.append(val)

            mlp_input = combined + level_emb

            # Forward through packed MLP (single fused matvec per layer)
            cv_in = dr.nn.CoopVec(*mlp_input)
            cv_in = dr.nn.cast(cv_in, Float16)
            cv_out = self.mlp(cv_in)
            correction = [mi.Float(f) for f in list(cv_out)]

            refined = [c + corr for c, corr in zip(combined, correction)]

            g_refined = refined[:self.num_param_per_gaussian]
            v_refined = refined[self.num_param_per_gaussian:]

            mean, variance = self.encoding_gaussian(g_refined)
            sharpness, axis, amplitude = self.encoding_vmf(v_refined)
            gaussians_list.append((mean, variance))
            vmfs_list.append((sharpness, axis, amplitude))

        return gaussians_list, vmfs_list
