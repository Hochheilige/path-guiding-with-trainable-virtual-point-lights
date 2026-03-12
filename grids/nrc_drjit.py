import math
import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from drjit.auto.ad import Float16, TensorXf16


# ─────────────────────────── encoding helpers ────────────────────────────────

def _triangle_wave(x: mi.Float) -> mi.Float:
    """Triangle wave approximation of sine: tri(x) = 2|x mod 2 - 1| - 1.
    Period 2, range [-1, 1].  Cheaper than sin() — no transcendental needed.
    Paper (Sec. 5): replaces sine in the frequency encoding."""
    x_mod2 = x - 2.0 * dr.floor(x * 0.5)   # x mod 2 ∈ [0, 2)
    return 2.0 * dr.abs(x_mod2 - 1.0) - 1.0


def encode_position_drjit(pos: mi.Point3f, freqs: int = 12) -> list:
    """Triangle-wave frequency encoding, sine-only (paper omits cosines).
    12 frequencies 2^0..2^11, one value per (freq, coord) → 3 × 12 = 36 floats.

    Phase offset -0.5 makes each feature antisymmetric (sine-like): tri(f*x - 0.5).
    Without this, tri(f*x) is cosine-like (even), so x=+a and x=-a produce
    identical features and the NRC cannot distinguish the red-wall side from
    the green-wall side of the Cornell box."""
    out = []
    for i in range(freqs):
        f = float(2 ** i)
        out.append(_triangle_wave(f * pos.x - 0.5))
        out.append(_triangle_wave(f * pos.y - 0.5))
        out.append(_triangle_wave(f * pos.z - 0.5))
    return out  # 36 values


def _one_blob_1d(x: mi.Float, k: int = 4) -> list:
    """Quartic one-blob encoding of a scalar x ∈ [0, 1] → k floats.
    Paper (Sec. 5 / Fig. 9): quartic(t) = 15/16*(1-t²)²  replaces the Gaussian kernel.
    k evenly-spaced centers in [0, 1]; kernel width = 1/(k-1)."""
    inv_sigma = float(k - 1)
    out = []
    for i in range(k):
        c     = float(i) / (k - 1)
        t     = (x - c) * inv_sigma
        inner = dr.maximum(mi.Float(0.0), 1.0 - t * t)   # clamp to 0 outside |t|>1
        val   = (15.0 / 16.0) * inner * inner
        out.append(val)
    return out


def encode_spherical_one_blob(vec: mi.Vector3f, k: int = 4) -> list:
    """Spherical-coordinate one-blob: ob(sph(ω)) → 2*k floats.
    Paper (Table 1): convert direction to (θ, φ) normalized to [0,1]²,
    then apply one-blob independently on each coordinate."""
    vec   = dr.normalize(vec)
    theta = dr.acos(dr.clamp(vec.z, mi.Float(-1.0 + 1e-6), mi.Float(1.0 - 1e-6))) \
            * (1.0 / math.pi)                          # polar angle   → [0, 1]
    phi   = dr.atan2(vec.y, vec.x) * (1.0 / (2.0 * math.pi)) + 0.5   # azimuth → [0, 1]
    return _one_blob_1d(theta, k) + _one_blob_1d(phi, k)              # 2*k values


def encode_roughness_drjit(alpha_u: mi.Float, alpha_v: mi.Float, k: int = 4) -> list:
    """One-blob on transformed roughness: ob(1 - exp(-r)) → k floats.
    Paper (Table 1): single scalar roughness mapped to [0,1) before encoding.
    Uses max(alpha_u, alpha_v) as isotropic approximation for anisotropic BSDFs."""
    r = dr.maximum(alpha_u, alpha_v)   # isotropic approximation
    t = 1.0 - dr.exp(-r)              # [0, ∞) → [0, 1)
    return _one_blob_1d(t, k)


# ─────────────────────────── NRC model ───────────────────────────────────────

class nrc_model_drjit:
    _is_drjit = True
    _is_nrc   = True

    # Input layout (Table 1, padded to 64 for TensorCore + implicit bias):
    #   36  position  (triangle-wave freq, 12 freqs × 3 coords)
    #    8  normal    (spherical one-blob, 2 × k=4)
    #    8  direction (spherical one-blob, 2 × k=4)
    #    4  roughness (one-blob, k=4)
    #    3  diffuse reflectance  (identity)
    #    3  specular reflectance (identity)
    #    2  padding = 1.0  (implicit bias in first layer, TensorCore alignment)
    #   ── total = 64
    INPUT_DIM  = 64
    N_NEURONS  = 64
    N_HIDDEN   = 3   # paper: 4 hidden layers = 1 (first) + 3 (N_HIDDEN) + 1 (output)
    OUTPUT_DIM = 3

    def __init__(self, config, bb_min=None, bb_max=None):
        self.config = config

        # Bounding box used to normalize world-space positions to [-1, 1].
        # Without normalization the triangle-wave encoding aliases badly for
        # scenes whose coordinates are not already in that range (e.g. kitchen).
        if bb_min is not None and bb_max is not None:
            self._bbox_min = mi.Point3f(bb_min)
            self._bbox_max = mi.Point3f(bb_max)
        else:
            # Fallback: assume Cornell-box-like [-1, 1] range
            self._bbox_min = mi.Point3f(-1.0)
            self._bbox_max = mi.Point3f( 1.0)

        # Build MLP: 64 → [64×ReLU] × 4 → 3  (no bias, as per NRC paper Table 1)
        mlp_alloc = dr.nn.Sequential(
            dr.nn.Linear(self.INPUT_DIM, self.N_NEURONS, bias=False), dr.nn.ReLU(),
            *(item for _ in range(self.N_HIDDEN)
              for item in [dr.nn.Linear(self.N_NEURONS, self.N_NEURONS, bias=False), dr.nn.ReLU()]),
            dr.nn.Linear(self.N_NEURONS, self.OUTPUT_DIM, bias=False)
        ).alloc(TensorXf16, self.INPUT_DIM)

        # Scale down ONLY the output layer so the MLP starts near zero output.
        # Scaling all layers causes Float16 signal vanishing through hidden layers
        # (outputs collapse to 0.0, blocking grads); only the output layer needs it.
        # Must be done before pack() — post-pack scatter breaks mlp's internal
        # AD reference to mlp_coeffs, making weight gradients zero.
        for layer in mlp_alloc.layers:
            if hasattr(layer, 'weights') and layer.weights.shape[0] == self.OUTPUT_DIM:
                layer.weights = TensorXf16(layer.weights * 0.01)

        self.mlp_coeffs, self.mlp = dr.nn.pack(mlp_alloc, layout='training')

        lr = config.optimizer.learning_rate
        self.opt = dr.opt.Adam(lr=lr)
        self.opt['nrc'] = mi.Float(self.mlp_coeffs.array)
        # Moderate loss scaling to prevent Float16 gradient underflow in the backward pass.
        # init_scale=1.0 was previously needed to avoid overflow with the too-deep 6-layer
        # network; with the correct 4-layer depth, 128 is safe and prevents underflow.
        self.scaler = dr.opt.GradScaler(init_scale=128.0)

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

    def _build_features(self, si: mi.SurfaceInteraction3f, ray: mi.Ray3f) -> tuple:
        """Assemble the 64-dim NRC feature vector (Table 1 + padding)."""
        bsdf = si.bsdf()
        alpha_u = bsdf.eval_attribute_1("alpha_u", si)
        alpha_v = bsdf.eval_attribute_1("alpha_v", si)

        diffuse  = bsdf.eval_diffuse_reflectance(si)
        specular = bsdf.eval_attribute("specular_reflectance", si)

        wo = dr.normalize(mi.Vector3f(-ray.d))

        n_rays = dr.width(si.p)

        # Normalize position to [-1, 1] so the frequency encoding works for
        # any scene scale, not just Cornell box's unit-cube coordinates.
        extent = self._bbox_max - self._bbox_min
        pos_norm = mi.Point3f(
            2.0 * (si.p.x - self._bbox_min.x) / extent.x - 1.0,
            2.0 * (si.p.y - self._bbox_min.y) / extent.y - 1.0,
            2.0 * (si.p.z - self._bbox_min.z) / extent.z - 1.0,
        )

        feat = (
            encode_position_drjit(pos_norm)                       # 36
            + encode_spherical_one_blob(mi.Vector3f(si.n))        #  8
            + encode_spherical_one_blob(wo)                       #  8
            + encode_roughness_drjit(alpha_u, alpha_v)            #  4
            + [diffuse.x,  diffuse.y,  diffuse.z]                 #  3
            + [specular.x, specular.y, specular.z]                #  3
            + [mi.Float(dr.full(mi.Float, 1.0, n_rays)),          #  2  padding = 1.0
               mi.Float(dr.full(mi.Float, 1.0, n_rays))]          #     (implicit bias)
        )                                                         # = 64

        return feat, diffuse, specular

    def __call__(self, si: mi.SurfaceInteraction3f, ray: mi.Ray3f, active=True):
        """Evaluate NRC: returns mi.Color3f indirect radiance."""
        active = mi.Bool(active) & si.is_valid()

        # Sync Float32 optimizer weights → Float16 MLP storage.
        # Must be INSIDE dr.resume_grad() so the scatter creates an AD edge:
        #   opt['nrc'] (Float32, Adam-tracked) → mlp_coeffs.array (Float16)
        # This is the correct pattern: scatter JUST before forward pass,
        # not at init time (init scatter would break mlp's internal AD reference).
        self.mlp_coeffs.array[:] = Float16(self.opt['nrc'])

        features, diffuse, specular = self._build_features(si, ray)

        # Guard: invalid si gives si.p = inf → triangle_wave(inf) = NaN.
        # NaN * 0 = NaN in IEEE 754, so masked lanes still corrupt gradients.
        features = [dr.select(dr.isfinite(f), f, mi.Float(0.0)) for f in features]
        dr.eval(*features)

        cv = dr.nn.CoopVec(*features)
        cv = dr.nn.cast(cv, Float16)
        cv_out = self.mlp(cv)

        out = [mi.Float(f) for f in list(cv_out)]
        dr.eval(*out)

        # Paper (Sec. 5 — Reflectance factorization):
        # "multiply the network output by the sum of the diffuse albedo and
        #  specular reflectance".  No output activation — the network learns
        #  to produce non-negative values through the training objective.
        refl = mi.Color3f(diffuse.x + specular.x,
                          diffuse.y + specular.y,
                          diffuse.z + specular.z)
        radiance = refl * mi.Color3f(out[0], out[1], out[2])

        return dr.select(active, radiance, mi.Color3f(0.0))
