import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from drjit.auto.ad import Float16, TensorXf16


# ─────────────────────────── encoding helpers ────────────────────────────────

def _freq_encode_scalar(x: mi.Float, freqs: int = 6):
    """sin/cos frequency encoding of a scalar → 2*freqs floats."""
    out = []
    for i in range(freqs):
        f = float(2 ** i)
        out.append(dr.sin(f * x))
        out.append(dr.cos(f * x))
    return out


def encode_position_drjit(pos: mi.Point3f, freqs: int = 6):
    """Frequency encode xyz → 3 * 2 * freqs = 36 floats."""
    return _freq_encode_scalar(pos.x, freqs) \
         + _freq_encode_scalar(pos.y, freqs) \
         + _freq_encode_scalar(pos.z, freqs)


def encode_one_blob_drjit(vec: mi.Vector3f, resolution: int = 8):
    """One-blob encoding of a direction (luminance-projected) → resolution floats."""
    vec = dr.normalize(vec)
    scalar = 0.2989 * vec.x + 0.5870 * vec.y + 0.1140 * vec.z
    centers = [-1.0 + 2.0 * k / (resolution - 1) for k in range(resolution)]
    return [dr.exp(-40.0 * (scalar - c) ** 2) for c in centers]


def encode_roughness_drjit(alpha_u: mi.Float, alpha_v: mi.Float, resolution: int = 4):
    """One-blob roughness encoding: max over u/v axis → resolution floats."""
    tu = 1.0 - dr.exp(-alpha_u)
    tv = 1.0 - dr.exp(-alpha_v)
    centers = [float(k) / (resolution - 1) for k in range(resolution)]
    return [dr.maximum(dr.exp(-40.0 * (tu - c) ** 2),
                       dr.exp(-40.0 * (tv - c) ** 2)) for c in centers]


# ─────────────────────────── NRC model ───────────────────────────────────────

class nrc_model_drjit:
    _is_drjit = True
    _is_nrc   = True

    # input layout:  36 (pos freq) + 8 (normal blob) + 8 (dir blob)
    #              +  4 (roughness) +  3 (diffuse)   + 3 (specular) = 62
    INPUT_DIM  = 62
    N_NEURONS  = 64
    N_HIDDEN   = 5
    OUTPUT_DIM = 3

    def __init__(self, config):
        self.config = config

        # Build MLP: 62 → [64×ReLU] × 5 → 3  (no bias, as per NRC paper)
        layers = [dr.nn.Linear(self.INPUT_DIM, self.N_NEURONS, bias=False), dr.nn.ReLU()]
        for _ in range(self.N_HIDDEN):
            layers += [dr.nn.Linear(self.N_NEURONS, self.N_NEURONS, bias=False), dr.nn.ReLU()]
        layers.append(dr.nn.Linear(self.N_NEURONS, self.OUTPUT_DIM, bias=False))

        mlp_alloc = dr.nn.Sequential(*layers).alloc(TensorXf16, self.INPUT_DIM)
        self.mlp_coeffs, self.mlp = dr.nn.pack(mlp_alloc, layout='training')

        lr = config.optimizer.learning_rate
        self.opt = dr.opt.Adam(lr=lr)
        self.opt['nrc'] = mi.Float(self.mlp_coeffs.array)
        self.scaler = dr.opt.GradScaler()

    @property
    def optimizer(self):
        return self

    def zero_grad(self):
        pass

    def step(self):
        self.scaler.step(self.opt)

    def set_current_epoch(self, epoch):
        pass

    def set_config(self, config):
        self.config = config

    def get_gaussians_for_debug_render(self):
        return [], []

    def _build_features(self, si: mi.SurfaceInteraction3f, ray: mi.Ray3f):
        """Assemble the 62-dim NRC feature vector.

        Returns (features: list[mi.Float], diffuse: mi.Color3f, specular: mi.Color3f)
        """
        bsdf = si.bsdf()
        alpha_u = bsdf.eval_attribute_1("alpha_u", si)
        alpha_v = bsdf.eval_attribute_1("alpha_v", si)

        diffuse  = bsdf.eval_diffuse_reflectance(si)
        specular = bsdf.eval_attribute("specular_reflectance", si)

        wo = dr.normalize(mi.Vector3f(-ray.d))

        feat = (
            encode_position_drjit(si.p)                      # 36
            + encode_one_blob_drjit(mi.Vector3f(si.n))       #  8
            + encode_one_blob_drjit(wo)                      #  8
            + encode_roughness_drjit(alpha_u, alpha_v)       #  4
            + [diffuse.x,   diffuse.y,   diffuse.z]          #  3
            + [specular.x,  specular.y,  specular.z]         #  3
        )                                                    # = 62

        return feat, diffuse, specular

    def __call__(self, si: mi.SurfaceInteraction3f, ray: mi.Ray3f, active=True):
        """Evaluate NRC: returns mi.Color3f indirect radiance."""
        active = mi.Bool(active) & si.is_valid()

        self.mlp_coeffs.array[:] = Float16(self.opt['nrc'])

        features, diffuse, specular = self._build_features(si, ray)

        cv = dr.nn.CoopVec(*[mi.Float(f) for f in features])
        cv = dr.nn.cast(cv, Float16)
        cv_out = self.mlp(cv)

        out = [mi.Float(f) for f in list(cv_out)]

        # relu(mlp_out) * (diffuse + specular)  — mirrors torch get_nrc_prediction
        scale = mi.Color3f(dr.maximum(out[0], 0.0),
                           dr.maximum(out[1], 0.0),
                           dr.maximum(out[2], 0.0))
        refl  = mi.Color3f(diffuse.x + specular.x,
                           diffuse.y + specular.y,
                           diffuse.z + specular.z)

        return dr.select(active, refl * scale, mi.Color3f(0.0))
