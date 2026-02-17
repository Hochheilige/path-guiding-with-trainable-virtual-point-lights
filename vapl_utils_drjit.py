import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import math
import numpy as np

# ---------------------------------------------------------------------------
# Helper functions (missing from DrJIT)
# ---------------------------------------------------------------------------

def dr_erfc(x):
    return 1.0 - dr.erf(x)

def dr_expm1(x):
    """exp(x) - 1 with small-x stability."""
    return dr.select(dr.abs(x) < 1e-7, x, dr.exp(x) - 1.0)

def dr_log1p(x):
    """log(1+x) with small-x stability."""
    return dr.select(dr.abs(x) < 1e-7, x, dr.log(1.0 + x))

def dr_sigmoid(x):
    return 1.0 / (1.0 + dr.exp(-x))

def dr_softplus(x):
    return dr.log(1.0 + dr.exp(x))

# ---------------------------------------------------------------------------
# SG / vMF math ported from vapl_utils.py
# ---------------------------------------------------------------------------

def sg_product(axis1, sharpness1, axis2, sharpness2):
    """Spherical Gaussian product.

    Args:
        axis1: mi.Vector3f
        sharpness1: mi.Float
        axis2: mi.Vector3f
        sharpness2: mi.Float
    Returns:
        (axis: Vector3f, sharpness: Float, log_amplitude: Float)
    """
    eps = 1e-7
    axis_vec = axis1 * sharpness1 + axis2 * sharpness2
    sharpness = dr.norm(axis_vec)

    d = axis1 - axis2
    len2 = dr.dot(d, d)

    denom = dr.maximum(sharpness + sharpness1 + sharpness2, eps)
    log_amplitude = -sharpness1 * sharpness2 * len2 / denom

    axis = axis_vec / dr.maximum(sharpness, eps)

    return axis, sharpness, log_amplitude


def upper_sg_clamp_cosine_integral_over_two_pi(sharpness):
    """[Tokuyoshi et al. 2024, Listing 5]"""
    # Polynomial for small sharpness (<= 0.5)
    s = sharpness
    poly = (((((((-1.0 / 362880.0) * s + 1.0 / 40320.0) * s - 1.0 / 5040.0) * s
               + 1.0 / 720.0) * s - 1.0 / 120.0) * s + 1.0 / 24.0) * s - 1.0 / 6.0) * s + 0.5

    # Rational for large sharpness
    rational = (dr_expm1(-s) + s) / (s * s)

    return dr.select(s <= 0.5, poly, rational)


def lower_sg_clamp_cosine_integral_over_two_pi(sharpness):
    """[Tokuyoshi et al. 2024, Listing 6]"""
    s = sharpness
    e = dr.exp(-s)

    poly = e * (((((((((1.0 / 403200.0) * s - 1.0 / 45360.0) * s
                      + 1.0 / 5760.0) * s - 1.0 / 840.0) * s
                    + 1.0 / 144.0) * s - 1.0 / 30.0) * s
                  + 1.0 / 8.0) * s - 1.0 / 3.0) * s + 0.5)

    rational = e * (-dr_expm1(-s) - s * e) / (s * s)

    return dr.select(s <= 0.5, poly, rational)


def sg_clamp_cosine_product_integral_over_pi(cosine, sharpness):
    """[Tokuyoshi et al. 2024, Listing 7]

    Args:
        cosine: mi.Float — dot(sg_axis, normal)
        sharpness: mi.Float
    Returns:
        mi.Float
    """
    A = 2.7360831611272558028247203765204
    B = 17.02129778174187535455530451145
    C = 4.0100826728510421403939290030394
    D = 15.219156263147210594866010069381
    E = 76.087896272360737270901154261082

    sqrt_term = 0.5 * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E)
    t = sharpness * dr.sqrt(sqrt_term)
    tz = t * cosine

    INV_SQRTPI = 0.56418958354775628694807945156077
    FLT_EPS = 1.1920928955078125e-07
    CLAMPING_THRESHOLD = 0.5 * FLT_EPS

    erfc_neg_tz = dr_erfc(-tz)
    erfc_t = dr_erfc(t)
    exp_neg_tz2 = dr.exp(-tz * tz)
    exp_safe = dr_expm1(t * t * (cosine * cosine - 1.0))

    exp_term = dr.select(dr.abs(t) > FLT_EPS, exp_safe / t, mi.Float(0.0))

    lerp_factor = dr.clamp(
        dr.maximum(
            0.5 * (cosine * erfc_neg_tz + erfc_t) - 0.5 * INV_SQRTPI * exp_neg_tz2 * exp_term,
            CLAMPING_THRESHOLD
        ),
        0.0, 1.0
    )

    lower_integral = lower_sg_clamp_cosine_integral_over_two_pi(sharpness)
    upper_integral = upper_sg_clamp_cosine_integral_over_two_pi(sharpness)

    return 2.0 * dr.lerp(lower_integral, upper_integral, lerp_factor)


def vmf_hemispherical_integral(cosine, sharpness):
    """[Tokuyoshi et al. 2024, Listing 4]

    Args:
        cosine: mi.Float — cos(angle between SG axis and hemisphere pole)
        sharpness: mi.Float
    Returns:
        mi.Float
    """
    A = 0.6517328826907056171791055021459
    B = 1.3418280033141287699294252888649
    C = 7.2216687798956709087860872386955
    steepness = sharpness * dr.sqrt((0.5 * sharpness + A) / ((sharpness + B) * sharpness + C))
    clamped_cos = dr.clamp(cosine, -1.0, 1.0)
    lerp_factor = dr.clamp(0.5 + 0.5 * (dr.erf(steepness * clamped_cos) / dr.erf(steepness)), 0.0, 1.0)

    e = dr.exp(-sharpness)
    return dr.lerp(e, mi.Float(1.0), lerp_factor) / (e + 1.0)


def expm1_over_x(x):
    """(exp(x) - 1)/x with numerical stability. [Higham, Section 1.14.1]"""
    u = dr.exp(x)
    y = u - 1.0
    close_to_zero = dr.abs(y) < 1e-6
    small_x = ~close_to_zero & (dr.abs(x) < 1.0)
    # For small_x: y / log(u), for large: y / x, for ~0: 1.0
    result_small = y / dr.log(u)
    result_large = y / x
    result = dr.select(close_to_zero, mi.Float(1.0),
             dr.select(small_x, result_small, result_large))
    return result


def sg_integral(sharpness):
    return 4.0 * math.pi * expm1_over_x(-2.0 * sharpness)


def compute_jacobian(wi_x, wi_y, wi_z):
    """Compute JJ^T as scalar 2x2 decomposition.

    Args:
        wi_x, wi_y, wi_z: mi.Float — view direction components in tangent space
    Returns:
        (jj00, jj01, jj10, jj11): mi.Float — entries of JJ^T matrix
    """
    vlen = dr.sqrt(wi_x * wi_x + wi_y * wi_y)
    safe_vlen = dr.maximum(vlen, 1e-8)

    vx = dr.select(vlen > 0.0, wi_x / safe_vlen, mi.Float(1.0))
    vy = dr.select(vlen > 0.0, wi_y / safe_vlen, mi.Float(0.0))

    # Jacobian matrix J = R * diag(0.5, 0.5/wi_z)
    # where R = [[vx, -vy], [vy, vx]]
    # J = [[0.5*vx, -0.5*vy/wi_z], [0.5*vy, 0.5*vx/wi_z]]
    safe_wz = dr.maximum(dr.abs(wi_z), 1e-8) * dr.sign(dr.select(wi_z == 0.0, mi.Float(1.0), wi_z))
    inv_wz = 1.0 / safe_wz

    j00 = 0.5 * vx
    j01 = -0.5 * vy * inv_wz
    j10 = 0.5 * vy
    j11 = 0.5 * vx * inv_wz

    # JJ^T = J * J^T
    jj00 = j00 * j00 + j01 * j01
    jj01 = j00 * j10 + j01 * j11
    jj10 = j10 * j00 + j11 * j01
    jj11 = j10 * j10 + j11 * j11

    return jj00, jj01, jj10, jj11


def isotropic_ndf_filtering(si):
    """Compute filtered roughness from surface interaction.

    Args:
        si: mi.SurfaceInteraction3f
    Returns:
        (roughness_u: mi.Float, roughness_v: mi.Float)
    """
    SIGMA2 = 0.15915494  # 1/(2*pi)
    KAPPA = 0.18

    dndu = si.dn_du
    dndv = si.dn_dv

    bsdf = si.bsdf()

    has_alpha = bsdf.has_attribute("alpha", active=True)
    has_alpha_u = bsdf.has_attribute("alpha_u", active=True)
    has_alpha_v = bsdf.has_attribute("alpha_v", active=True)

    alpha_mask = has_alpha & ~has_alpha_u & ~has_alpha_v
    alpha_uv_mask = has_alpha_u & has_alpha_v

    n = dr.width(si.p)
    alpha_u = dr.zeros(mi.Float, n)
    alpha_v = dr.zeros(mi.Float, n)

    if dr.any(alpha_mask):
        alpha_val = bsdf.eval_attribute_1("alpha", si, active=alpha_mask)
        alpha_u = dr.select(alpha_mask, alpha_val, alpha_u)
        alpha_v = dr.select(alpha_mask, alpha_val, alpha_v)

    if dr.any(alpha_uv_mask):
        alpha_u_val = bsdf.eval_attribute_1("alpha_u", si, active=alpha_uv_mask)
        alpha_v_val = bsdf.eval_attribute_1("alpha_v", si, active=alpha_uv_mask)
        alpha_u = dr.select(alpha_uv_mask, alpha_u_val, alpha_u)
        alpha_v = dr.select(alpha_uv_mask, alpha_v_val, alpha_v)

    # kernel roughness from Eq.14
    kernel_roughness2 = SIGMA2 * (dr.dot(dndu, dndu) + dr.dot(dndv, dndv))
    clamped_kernel_roughness2 = dr.minimum(kernel_roughness2, KAPPA)

    # Filtered roughness
    eps = 1e-8
    filtered_ru2 = dr.clamp(alpha_u * alpha_u + clamped_kernel_roughness2, 0.0, 1.0)
    filtered_rv2 = dr.clamp(alpha_v * alpha_v + clamped_kernel_roughness2, 0.0, 1.0)

    return dr.sqrt(filtered_ru2), dr.sqrt(filtered_rv2)


def sggx(m_x, m_y, m_z, r00, r01, r10, r11):
    """SGGX NDF evaluation with scalar 2x2 roughness matrix.

    Args:
        m_x, m_y, m_z: mi.Float — half-vector components
        r00, r01, r10, r11: mi.Float — 2x2 roughness matrix entries
    Returns:
        mi.Float
    """
    eps = 1e-4

    # det of 2x2 roughness matrix
    det = r00 * r11 - r01 * r10
    det = dr.maximum(det, eps)

    # adjugate of 2x2 matrix
    adj00 = r11
    adj01 = -r01
    adj10 = -r10
    adj11 = r00

    # m_xy^T * adj * m_xy / det
    # = (m_x * adj00 + m_y * adj10) * m_x + (m_x * adj01 + m_y * adj11) * m_y
    term1_x = m_x * adj00 + m_y * adj10
    term1_y = m_x * adj01 + m_y * adj11
    quad = (term1_x * m_x + term1_y * m_y) / det

    length2 = quad + m_z * m_z
    length2 = dr.maximum(length2, eps)

    sqrt_det = dr.maximum(dr.sqrt(det), eps)
    denom = sqrt_det * length2 * length2

    return 1.0 / (math.pi * denom)


def sgg_reflection_pdf(wi_x, wi_y, wi_z, m_x, m_y, m_z, r00, r01, r10, r11):
    """SGGX reflection PDF.

    Args:
        wi_x, wi_y, wi_z: mi.Float — view direction in tangent space
        m_x, m_y, m_z: mi.Float — half-vector in tangent space
        r00, r01, r10, r11: mi.Float — roughness matrix entries
    Returns:
        mi.Float
    """
    # rough_wi = roughness_mat @ wi_xy
    rw_x = r00 * wi_x + r01 * wi_y
    rw_y = r10 * wi_x + r11 * wi_y

    # sqrt(wi_xy . rough_wi + wi_z^2)
    denom = dr.sqrt(wi_x * rw_x + wi_y * rw_y + wi_z * wi_z)
    denom = dr.maximum(denom, 1e-8)

    ndf = sggx(m_x, m_y, m_z, r00, r01, r10, r11)
    return ndf / (4.0 * denom)


def sample_vmf(axis, sharpness):
    """Sample a direction from a von Mises-Fisher distribution.

    Args:
        axis: mi.Vector3f — distribution axis
        sharpness: mi.Float — concentration parameter
    Returns:
        mi.Vector3f — sampled direction
    """
    n = dr.width(axis)
    rand0 = mi.Float(dr.arange(mi.Float, n))  # placeholder for sampler
    rand0 = mi.Float(np.random.uniform(size=n).astype(np.float32))
    rand1 = mi.Float(np.random.uniform(size=n).astype(np.float32))

    phi = 2.0 * math.pi * rand0
    THRESHOLD = 1.1920928955078125e-07 / 4.0

    r = dr.select(
        sharpness > THRESHOLD,
        dr_log1p(rand1 * dr_expm1(-2.0 * sharpness)) / sharpness,
        -2.0 * rand1
    )

    cos_theta = 1.0 + r
    sin_theta = dr.sqrt(dr.maximum(-r * r - 2.0 * r, 0.0))

    local_dir = mi.Vector3f(
        dr.cos(phi) * sin_theta,
        dr.sin(phi) * sin_theta,
        cos_theta
    )

    # Transform from local frame (z-axis aligned) to world frame aligned with axis
    frame = mi.Frame3f(axis)
    return frame.to_world(local_dir)


# ---------------------------------------------------------------------------
# BSDF approximation helpers
# ---------------------------------------------------------------------------

class cosine_lobe_sg_drjit:
    """Diffuse BSDF approximated as a cosine-lobe SG."""
    def __init__(self, normal):
        """
        Args:
            normal: mi.Vector3f — surface normal
        """
        self.axis = normal
        n = dr.width(normal)
        self.sharpness = mi.Float(dr.full(mi.Float, 2.123, n))
        self.amplitude = mi.Float(dr.full(mi.Float, 1.17, n))


# ---------------------------------------------------------------------------
# vapl_mixture_drjit
# ---------------------------------------------------------------------------

class vapl_mixture_drjit:
    """VAPL mixture with all-DrJIT convolution. No torch conversions."""

    def __init__(self, gaussians_list, vmfs_list, sweep_encoding):
        """
        Args:
            gaussians_list: list of (mean: Point3f, variance: Float)
            vmfs_list: list of (sharpness: Float, axis: Vector3f, amplitude: Color3f)
            sweep_encoding: str — axis encoding mode
        """
        self.mean = [g[0] for g in gaussians_list]
        self.variance = [g[1] for g in gaussians_list]
        self.sharpness = [v[0] for v in vmfs_list]
        self.axis = [v[1] for v in vmfs_list]
        self.amplitude = [v[2] for v in vmfs_list]

        n = dr.width(gaussians_list[0][0])
        self.illumination = mi.Color3f(0.0)
        self.diffuse_illumination = mi.Color3f(0.0)
        self.specular_illumination = mi.Color3f(0.0)

    # def _compute_importance_score(self, index, si):
    #     """Compute a cheap importance score for VAPL at given index.
    #
    #     Uses the SG product log-amplitude between the VAPL's directional
    #     distribution and the light direction from the shading point.
    #     This measures how well the VAPL "points toward" the shading point.
    #
    #     Args:
    #         index: int — which VAPL in the mixture
    #         si: mi.SurfaceInteraction3f
    #     Returns:
    #         mi.Float — per-ray importance score (non-negative)
    #     """
    #     SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
    #     eps = 1e-4
    #
    #     light_vec = self.mean[index] - si.p
    #     squared_distance = dr.dot(light_vec, light_vec)
    #     light_dir = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))
    #
    #     variance = dr.maximum(self.variance[index], squared_distance / SGLIGHT_SHARPNESS_MAX)
    #     light_sharpness = squared_distance / dr.maximum(variance, eps)
    #
    #     _, _, log_amplitude = sg_product(
    #         self.axis[index], self.sharpness[index], light_dir, light_sharpness)
    #
    #     return dr.exp(log_amplitude)

    def convolve(self, si, view_dir):
        """Convolve all VAPLs with BSDF in a single batched evaluation.

        Instead of looping over N VAPLs and calling convolution N times,
        this concatenates all VAPL parameters into arrays of size N*R
        (N VAPLs, R rays), tiles the surface quantities to match, runs
        the convolution math once, then sums results back to R rays.

        Args:
            si: mi.SurfaceInteraction3f
            view_dir: mi.Vector3f — ray direction
        """
        N = len(self.mean)
        if N == 0:
            return

        # Fallback to simple loop for single VAPL (no benefit from batching)
        if N == 1:
            self._convolve_single(0, si, view_dir)
            return

        SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
        eps = 1e-4
        R = dr.width(si.p)
        total = N * R

        # --- Step 1: Compute surface-dependent quantities ONCE (R elements) ---
        position = si.p
        normal = si.n
        wo_world = dr.normalize(-view_dir)
        wi_world = dr.normalize(view_dir)
        wo_local = si.sh_frame.to_local(wo_world)
        wi_local = si.sh_frame.to_local(wi_world)

        bsdf = si.bsdf()

        ctx_diffuse = mi.BSDFContext()
        ctx_diffuse.type_mask = mi.BSDFFlags.Diffuse
        diffuse = bsdf.eval(ctx_diffuse, si, wo_local)

        wi_x = wi_local.x
        wi_y = wi_local.y
        wi_z = wi_local.z
        wi_z_safe = dr.select(wi_z == 0.0, mi.Float(eps), wi_z)

        jj00, jj01, jj10, jj11 = compute_jacobian(wi_x, wi_y, wi_z_safe)
        det_jj4 = 1.0 / (4.0 * wi_z_safe * wi_z_safe)

        roughness_u, roughness_v = isotropic_ndf_filtering(si)
        roughness2_u = roughness_u * roughness_u
        roughness2_v = roughness_v * roughness_v
        proj_roughness2_u = roughness2_u / dr.maximum(1.0 - roughness2_u, eps)
        proj_roughness2_v = roughness2_v / dr.maximum(1.0 - roughness2_v, eps)
        roughness_max2 = dr.maximum(roughness2_u, roughness2_v)

        reflect_sharpness = (1.0 - roughness_max2) / dr.maximum(2.0 * roughness_max2, eps)
        reflect_vec = mi.reflect(wo_world, normal)
        reflect_vec_scaled = reflect_vec * reflect_sharpness

        has_glossy = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Glossy)
        has_specular_reflectance = bsdf.has_attribute("specular_reflectance", active=True)
        specular_reflectance = dr.select(
            has_glossy,
            dr.select(has_specular_reflectance,
                      bsdf.eval_attribute_3("specular_reflectance", si, active=has_specular_reflectance),
                      mi.Spectrum(1.0)),
            mi.Spectrum(0.0)
        )

        wo_local_x = wo_local.x
        wo_local_y = wo_local.y
        wo_local_z = wo_local.z

        # Materialize surface quantities before tiling
        dr.eval(position, normal, diffuse, wo_local, wi_local,
                jj00, jj01, jj10, jj11, det_jj4,
                proj_roughness2_u, proj_roughness2_v,
                reflect_vec_scaled, specular_reflectance)

        # --- Step 2: Tile surface quantities N times (R -> N*R) ---
        # Index: for each of N*R elements, maps to the ray index in [0, R)
        tile_idx = dr.arange(mi.UInt32, total) % R

        # Helper: gather a Float array by tile_idx
        def tile_f(arr):
            return dr.gather(mi.Float, arr, tile_idx)

        # Tile scalar quantities
        t_pos_x = tile_f(position.x); t_pos_y = tile_f(position.y); t_pos_z = tile_f(position.z)
        t_position = mi.Point3f(t_pos_x, t_pos_y, t_pos_z)
        t_normal = mi.Vector3f(tile_f(normal.x), tile_f(normal.y), tile_f(normal.z))
        t_diffuse = mi.Spectrum(tile_f(diffuse.x), tile_f(diffuse.y), tile_f(diffuse.z))
        t_jj00 = tile_f(jj00)
        t_jj01 = tile_f(jj01)
        t_jj10 = tile_f(jj10)
        t_jj11 = tile_f(jj11)
        t_det_jj4 = tile_f(det_jj4)
        t_proj_roughness2_u = tile_f(proj_roughness2_u)
        t_proj_roughness2_v = tile_f(proj_roughness2_v)
        t_reflect_vec_scaled = mi.Vector3f(tile_f(reflect_vec_scaled.x), tile_f(reflect_vec_scaled.y), tile_f(reflect_vec_scaled.z))
        t_specular_reflectance = mi.Spectrum(tile_f(specular_reflectance.x), tile_f(specular_reflectance.y), tile_f(specular_reflectance.z))
        t_wo_local_x = tile_f(wo_local_x)
        t_wo_local_y = tile_f(wo_local_y)
        t_wo_local_z = tile_f(wo_local_z)

        # Tile shading frame vectors for manual to_local later
        sh_s = si.sh_frame.s
        sh_t = si.sh_frame.t
        sh_n = si.sh_frame.n
        dr.eval(sh_s, sh_t, sh_n)
        t_sh_s = mi.Vector3f(tile_f(sh_s.x), tile_f(sh_s.y), tile_f(sh_s.z))
        t_sh_t = mi.Vector3f(tile_f(sh_t.x), tile_f(sh_t.y), tile_f(sh_t.z))
        t_sh_n = mi.Vector3f(tile_f(sh_n.x), tile_f(sh_n.y), tile_f(sh_n.z))

        # --- Step 3: Concatenate all VAPL parameters (N*R) ---
        # Helper: scatter into a Float array
        b_mean_x = dr.zeros(mi.Float, total); b_mean_y = dr.zeros(mi.Float, total); b_mean_z = dr.zeros(mi.Float, total)
        b_variance = dr.zeros(mi.Float, total)
        b_sharpness = dr.zeros(mi.Float, total)
        b_axis_x = dr.zeros(mi.Float, total); b_axis_y = dr.zeros(mi.Float, total); b_axis_z = dr.zeros(mi.Float, total)
        b_amp_x = dr.zeros(mi.Float, total); b_amp_y = dr.zeros(mi.Float, total); b_amp_z = dr.zeros(mi.Float, total)

        for i in range(N):
            idx = dr.arange(mi.UInt32, R) + i * R
            dr.scatter(b_mean_x, self.mean[i].x, idx)
            dr.scatter(b_mean_y, self.mean[i].y, idx)
            dr.scatter(b_mean_z, self.mean[i].z, idx)
            dr.scatter(b_variance, self.variance[i], idx)
            dr.scatter(b_sharpness, self.sharpness[i], idx)
            dr.scatter(b_axis_x, self.axis[i].x, idx)
            dr.scatter(b_axis_y, self.axis[i].y, idx)
            dr.scatter(b_axis_z, self.axis[i].z, idx)
            dr.scatter(b_amp_x, self.amplitude[i].x, idx)
            dr.scatter(b_amp_y, self.amplitude[i].y, idx)
            dr.scatter(b_amp_z, self.amplitude[i].z, idx)

        b_mean = mi.Point3f(b_mean_x, b_mean_y, b_mean_z)
        b_axis = mi.Vector3f(b_axis_x, b_axis_y, b_axis_z)
        b_amplitude = mi.Color3f(b_amp_x, b_amp_y, b_amp_z)

        # --- Step 4: Batched VAPL-BSDF convolution (N*R elements) ---
        light_vec = b_mean - t_position
        squared_distance = dr.dot(light_vec, light_vec)
        light_dir = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))

        b_variance = dr.maximum(b_variance, squared_distance / SGLIGHT_SHARPNESS_MAX)
        inv_variance = 1.0 / dr.maximum(b_variance, eps)
        emissive = b_amplitude * inv_variance
        light_sharpness = squared_distance / dr.maximum(b_variance, eps)

        light_lobe_axis, light_lobe_sharpness, light_lobe_log_amplitude = sg_product(
            b_axis, b_sharpness, light_dir, light_sharpness)

        amp = dr.exp(light_lobe_log_amplitude)
        cosine = dr.clamp(dr.dot(light_lobe_axis, t_normal), -1.0, 1.0)
        diffuse_integral = sg_clamp_cosine_product_integral_over_pi(cosine, light_lobe_sharpness)
        b_diffuse_illum = t_diffuse * (amp * diffuse_integral)

        # Glossy SG lighting
        prod_vec = t_reflect_vec_scaled + light_lobe_axis * light_lobe_sharpness
        prod_sharpness = dr.norm(prod_vec)
        prod_dir = prod_vec / dr.maximum(prod_sharpness, eps)

        light_lobe_variance = 1.0 / dr.maximum(light_lobe_sharpness, eps)
        doubled_var = 2.0 * light_lobe_variance

        fr00 = t_proj_roughness2_u + doubled_var * t_jj00
        fr01 = doubled_var * t_jj01
        fr10 = doubled_var * t_jj10
        fr11 = t_proj_roughness2_v + doubled_var * t_jj11

        det = (t_proj_roughness2_u * t_proj_roughness2_v
               + 2.0 * light_lobe_variance * (t_proj_roughness2_u * t_jj00 + t_proj_roughness2_v * t_jj11)
               + light_lobe_variance * light_lobe_variance * t_det_jj4)

        tr = fr00 + fr11
        denom_filt = 1.0 + tr + det
        is_finite = dr.isfinite(denom_filt)

        filtered_r00 = dr.select(is_finite,
                                 dr.minimum(fr00 + det, 3.4028235e+38) / denom_filt,
                                 dr.minimum(fr00, 3.4028235e+38) / dr.minimum(fr00 + 1.0, 3.4028235e+38))
        filtered_r01 = dr.select(is_finite, fr01 / denom_filt, mi.Float(0.0))
        filtered_r10 = dr.select(is_finite, fr10 / denom_filt, mi.Float(0.0))
        filtered_r11 = dr.select(is_finite,
                                 dr.minimum(fr11 + det, 3.4028235e+38) / denom_filt,
                                 dr.minimum(fr11, 3.4028235e+38) / dr.minimum(fr11 + 1.0, 3.4028235e+38))

        visibility = vmf_hemispherical_integral(dr.dot(prod_dir, t_normal), prod_sharpness)

        # Half vector in tangent space (manual to_local using tiled frame)
        ll_local_x = dr.dot(light_lobe_axis, t_sh_s)
        ll_local_y = dr.dot(light_lobe_axis, t_sh_t)
        ll_local_z = dr.dot(light_lobe_axis, t_sh_n)
        hv_x = t_wo_local_x + ll_local_x
        hv_y = t_wo_local_y + ll_local_y
        hv_z = t_wo_local_z + ll_local_z
        hv_norm = dr.rsqrt(dr.maximum(hv_x * hv_x + hv_y * hv_y + hv_z * hv_z, eps))
        hv_x = hv_x * hv_norm
        hv_y = hv_y * hv_norm
        hv_z = hv_z * hv_norm

        lobe = sgg_reflection_pdf(t_wo_local_x, t_wo_local_y, t_wo_local_z,
                                  hv_x, hv_y, hv_z,
                                  filtered_r00, filtered_r01, filtered_r10, filtered_r11)

        sg_int = sg_integral(light_lobe_sharpness)

        b_specular_illum = t_specular_reflectance * (amp * visibility * lobe * sg_int)
        b_result = emissive * (b_diffuse_illum + b_specular_illum)

        # --- Step 5: Sum across N VAPLs back to R rays ---
        # scatter_reduce component-wise (struct types need flat 1D arrays)
        sum_diff_x = dr.zeros(mi.Float, R); sum_diff_y = dr.zeros(mi.Float, R); sum_diff_z = dr.zeros(mi.Float, R)
        sum_spec_x = dr.zeros(mi.Float, R); sum_spec_y = dr.zeros(mi.Float, R); sum_spec_z = dr.zeros(mi.Float, R)
        sum_res_x = dr.zeros(mi.Float, R); sum_res_y = dr.zeros(mi.Float, R); sum_res_z = dr.zeros(mi.Float, R)

        dr.scatter_reduce(dr.ReduceOp.Add, sum_diff_x, b_diffuse_illum.x, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_diff_y, b_diffuse_illum.y, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_diff_z, b_diffuse_illum.z, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_spec_x, b_specular_illum.x, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_spec_y, b_specular_illum.y, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_spec_z, b_specular_illum.z, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_res_x, b_result.x, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_res_y, b_result.y, tile_idx)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_res_z, b_result.z, tile_idx)

        self.diffuse_illumination = self.diffuse_illumination + mi.Color3f(sum_diff_x, sum_diff_y, sum_diff_z)
        self.specular_illumination = self.specular_illumination + mi.Color3f(sum_spec_x, sum_spec_y, sum_spec_z)
        self.illumination = self.illumination + mi.Color3f(sum_res_x, sum_res_y, sum_res_z)

    def _convolve_single(self, index, si, view_dir):
        """Convolve a single VAPL (by index) with BSDF. Used as fallback for N=1."""
        SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
        eps = 1e-4

        mean = self.mean[index]
        variance = self.variance[index]
        sharpness = self.sharpness[index]
        axis = self.axis[index]
        amplitude = self.amplitude[index]

        position = si.p
        normal = si.n

        wo_world = dr.normalize(-view_dir)
        wi_world = dr.normalize(view_dir)

        wo_local = si.sh_frame.to_local(wo_world)
        wi_local = si.sh_frame.to_local(wi_world)

        bsdf = si.bsdf()

        # Light vector from shading point to VAPL mean
        light_vec = mean - position
        squared_distance = dr.dot(light_vec, light_vec)
        light_dir = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))

        # Clamp variance for numerical stability
        variance = dr.maximum(variance, squared_distance / SGLIGHT_SHARPNESS_MAX)

        # Maximum emissive radiance of the VAPL
        inv_variance = 1.0 / dr.maximum(variance, eps)
        emissive = amplitude * inv_variance

        # VAPL sharpness for the light distribution viewed from shading point
        light_sharpness = squared_distance / dr.maximum(variance, eps)

        # Light lobe = product of light distribution and directional distribution
        light_lobe_axis, light_lobe_sharpness, light_lobe_log_amplitude = sg_product(
            axis, sharpness, light_dir, light_sharpness)

        # --- Diffuse SG lighting ---
        # [Tokuyoshi et al. 2024, Section 4]
        ctx_diffuse = mi.BSDFContext()
        ctx_diffuse.type_mask = mi.BSDFFlags.Diffuse
        diffuse = bsdf.eval(ctx_diffuse, si, wo_local)

        amp = dr.exp(light_lobe_log_amplitude)
        cosine = dr.clamp(dr.dot(light_lobe_axis, normal), -1.0, 1.0)

        diffuse_integral = sg_clamp_cosine_product_integral_over_pi(cosine, light_lobe_sharpness)
        diffuse_illumination = diffuse * (amp * diffuse_integral)

        # --- Glossy SG lighting ---
        # [Tokuyoshi et al. 2024, Section 5]

        # Compute JJ^T for NDF filtering
        wi_x = wi_local.x
        wi_y = wi_local.y
        wi_z = wi_local.z

        wi_z_safe = dr.select(wi_z == 0.0, mi.Float(eps), wi_z)

        jj00, jj01, jj10, jj11 = compute_jacobian(wi_x, wi_y, wi_z_safe)

        det_jj4 = 1.0 / (4.0 * wi_z_safe * wi_z_safe)

        # NDF filtering
        roughness_u, roughness_v = isotropic_ndf_filtering(si)
        roughness2_u = roughness_u * roughness_u
        roughness2_v = roughness_v * roughness_v

        proj_roughness2_u = roughness2_u / dr.maximum(1.0 - roughness2_u, eps)
        proj_roughness2_v = roughness2_v / dr.maximum(1.0 - roughness2_v, eps)

        roughness_max2 = dr.maximum(roughness2_u, roughness2_v)

        reflect_sharpness = (1.0 - roughness_max2) / dr.maximum(2.0 * roughness_max2, eps)
        reflect_vec = mi.reflect(wo_world, normal)
        reflect_vec_scaled = reflect_vec * reflect_sharpness

        # Product of reflection lobe and light lobe
        prod_vec = reflect_vec_scaled + light_lobe_axis * light_lobe_sharpness
        prod_sharpness = dr.norm(prod_vec)
        prod_dir = prod_vec / dr.maximum(prod_sharpness, eps)

        light_lobe_variance = 1.0 / dr.maximum(light_lobe_sharpness, eps)

        # Filtered projected roughness matrix + light lobe variance * JJ^T
        doubled_var = 2.0 * light_lobe_variance
        fr00 = proj_roughness2_u + doubled_var * jj00
        fr01 = doubled_var * jj01
        fr10 = doubled_var * jj10
        fr11 = proj_roughness2_v + doubled_var * jj11

        # Determinant computation (numerically stable, Supplementary Section 5.2)
        det = (proj_roughness2_u * proj_roughness2_v
               + 2.0 * light_lobe_variance * (proj_roughness2_u * jj00 + proj_roughness2_v * jj11)
               + light_lobe_variance * light_lobe_variance * det_jj4)

        # NDF filtering (Supplementary Section 5.2)
        tr = fr00 + fr11
        denom_filt = 1.0 + tr + det
        is_finite = dr.isfinite(denom_filt)

        # Finite case: (filtered + det*I) / (1 + tr + det)
        filtered_r00 = dr.select(is_finite,
                                 dr.minimum(fr00 + det, 3.4028235e+38) / denom_filt,
                                 dr.minimum(fr00, 3.4028235e+38) / dr.minimum(fr00 + 1.0, 3.4028235e+38))
        filtered_r01 = dr.select(is_finite, fr01 / denom_filt, mi.Float(0.0))
        filtered_r10 = dr.select(is_finite, fr10 / denom_filt, mi.Float(0.0))
        filtered_r11 = dr.select(is_finite,
                                 dr.minimum(fr11 + det, 3.4028235e+38) / denom_filt,
                                 dr.minimum(fr11, 3.4028235e+38) / dr.minimum(fr11 + 1.0, 3.4028235e+38))

        # Visibility of the SG light in the upper hemisphere
        visibility = vmf_hemispherical_integral(dr.dot(prod_dir, normal), prod_sharpness)

        # Evaluate filtered reflection lobe
        wo_local_x = wo_local.x
        wo_local_y = wo_local.y
        wo_local_z = wo_local.z

        # Half vector in tangent space
        light_lobe_axis_local = si.sh_frame.to_local(light_lobe_axis)
        half_vec_unnorm = mi.Vector3f(wo_local_x, wo_local_y, wo_local_z) + light_lobe_axis_local
        half_vec = dr.normalize(half_vec_unnorm)

        lobe = sgg_reflection_pdf(wo_local_x, wo_local_y, wo_local_z,
                                  half_vec.x, half_vec.y, half_vec.z,
                                  filtered_r00, filtered_r01, filtered_r10, filtered_r11)

        sg_int = sg_integral(light_lobe_sharpness)

        # Specular reflectance
        has_glossy = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Glossy)
        has_specular_reflectance = bsdf.has_attribute("specular_reflectance", active=True)
        specular_reflectance = dr.select(
            has_glossy,
            dr.select(has_specular_reflectance,
                      bsdf.eval_attribute_3("specular_reflectance", si, active=has_specular_reflectance),
                      mi.Spectrum(1.0)),
            mi.Spectrum(0.0)
        )

        specular_illumination = specular_reflectance * (amp * visibility * lobe * sg_int)

        result = emissive * (diffuse_illumination + specular_illumination)

        # Accumulate
        self.diffuse_illumination = self.diffuse_illumination + diffuse_illumination
        self.specular_illumination = self.specular_illumination + specular_illumination
        self.illumination = self.illumination + result
