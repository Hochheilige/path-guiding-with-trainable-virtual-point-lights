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
# SG / vMF math
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
    s = sharpness
    poly = (((((((-1.0 / 362880.0) * s + 1.0 / 40320.0) * s - 1.0 / 5040.0) * s
               + 1.0 / 720.0) * s - 1.0 / 120.0) * s + 1.0 / 24.0) * s - 1.0 / 6.0) * s + 0.5

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

    safe_wz = dr.maximum(dr.abs(wi_z), 1e-8) * dr.sign(dr.select(wi_z == 0.0, mi.Float(1.0), wi_z))
    inv_wz = 1.0 / safe_wz

    j00 = 0.5 * vx
    j01 = -0.5 * vy * inv_wz
    j10 = 0.5 * vy
    j11 = 0.5 * vx * inv_wz

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

    kernel_roughness2 = SIGMA2 * (dr.dot(dndu, dndu) + dr.dot(dndv, dndv))
    clamped_kernel_roughness2 = dr.minimum(kernel_roughness2, KAPPA)

    eps = 1e-8
    filtered_ru2 = dr.clamp(alpha_u * alpha_u + clamped_kernel_roughness2, 0.0, 1.0)
    filtered_rv2 = dr.clamp(alpha_v * alpha_v + clamped_kernel_roughness2, 0.0, 1.0)

    return dr.sqrt(filtered_ru2), dr.sqrt(filtered_rv2)


def sggx(m_x, m_y, m_z, r00, r01, r10, r11):
    """SGGX NDF evaluation with scalar 2x2 roughness matrix."""
    eps = 1e-4

    det = r00 * r11 - r01 * r10
    det = dr.maximum(det, eps)

    adj00 = r11
    adj01 = -r01
    adj10 = -r10
    adj11 = r00

    term1_x = m_x * adj00 + m_y * adj10
    term1_y = m_x * adj01 + m_y * adj11
    quad = (term1_x * m_x + term1_y * m_y) / det

    length2 = quad + m_z * m_z
    length2 = dr.maximum(length2, eps)

    sqrt_det = dr.maximum(dr.sqrt(det), eps)
    denom = sqrt_det * length2 * length2

    return 1.0 / (math.pi * denom)


def sgg_reflection_pdf(wi_x, wi_y, wi_z, m_x, m_y, m_z, r00, r01, r10, r11):
    """SGGX reflection PDF."""
    rw_x = r00 * wi_x + r01 * wi_y
    rw_y = r10 * wi_x + r11 * wi_y

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
