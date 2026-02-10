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

    def convolve(self, si, view_dir):
        for i in range(len(self.mean)):
            self.convolve_with_bsdf(i, si, view_dir)

    def convolve_with_bsdf(self, index, si, view_dir):
        """All-DrJIT convolution of VAPL with BSDF at surface intersection.

        Args:
            index: int — which VAPL in the mixture
            si: mi.SurfaceInteraction3f
            view_dir: mi.Vector3f — ray direction (from camera toward surface)
        """
        SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
        eps = 1e-4

        position = si.p
        normal = si.n

        # Direction conventions (matching original PyTorch code):
        #   view_dir = ray.d = from camera toward surface
        #   wo_world = from surface to camera (for bsdf.eval, half vector, reflection)
        #   wi_world = from camera to surface (for jacobian/NDF filtering)
        wo_world = dr.normalize(-view_dir)   # from surface to camera
        wi_world = dr.normalize(view_dir)    # from camera to surface

        wo_local = si.sh_frame.to_local(wo_world)   # for bsdf.eval + half vector
        wi_local = si.sh_frame.to_local(wi_world)   # for jacobian/NDF

        bsdf = si.bsdf()

        # Light vector from shading point to VAPL mean
        light_vec = self.mean[index] - position
        squared_distance = dr.dot(light_vec, light_vec)
        light_dir = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))

        # Clamp variance for numerical stability
        variance = dr.maximum(self.variance[index], squared_distance / SGLIGHT_SHARPNESS_MAX)

        # Maximum emissive radiance of the VAPL
        inv_variance = 1.0 / dr.maximum(variance, eps)
        emissive = self.amplitude[index] * inv_variance

        # VAPL sharpness for the light distribution viewed from shading point
        light_sharpness = squared_distance / dr.maximum(variance, eps)

        # Light lobe = product of light distribution and directional distribution
        light_lobe_axis, light_lobe_sharpness, light_lobe_log_amplitude = sg_product(
            self.axis[index], self.sharpness[index], light_dir, light_sharpness)

        # --- Diffuse SG lighting ---
        # [Tokuyoshi et al. 2024, Section 4]
        ctx_diffuse = mi.BSDFContext()
        ctx_diffuse.type_mask = mi.BSDFFlags.Diffuse
        diffuse = bsdf.eval(ctx_diffuse, si, wo_local)

        amplitude = dr.exp(light_lobe_log_amplitude)
        cosine = dr.clamp(dr.dot(light_lobe_axis, normal), -1.0, 1.0)

        diffuse_integral = sg_clamp_cosine_product_integral_over_pi(cosine, light_lobe_sharpness)
        diffuse_illumination = diffuse * (amplitude * diffuse_integral)

        # --- Glossy SG lighting ---
        # [Tokuyoshi et al. 2024, Section 5]

        # Compute JJ^T for NDF filtering
        wi_x = wi_local.x
        wi_y = wi_local.y
        wi_z = wi_local.z

        # Avoid division by zero in wi_z
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

        specular_illumination = specular_reflectance * (amplitude * visibility * lobe * sg_int)

        result = emissive * (diffuse_illumination + specular_illumination)

        # Accumulate
        self.diffuse_illumination = self.diffuse_illumination + diffuse_illumination
        self.specular_illumination = self.specular_illumination + specular_illumination
        self.illumination = self.illumination + result
