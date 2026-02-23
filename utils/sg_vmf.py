import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import math
import torch
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


def spherical(tensor_list):
    result = []
    for x in tensor_list:
        theta = torch.sigmoid(x[:, 0])
        phi = torch.sigmoid(x[:, 1])
        axis = torch.stack([
            torch.sin(theta) * torch.cos(phi),
            torch.sin(theta) * torch.sin(phi),
            torch.cos(theta)
        ], dim=1)  # (N, 3)
        result.append(axis)
    return result

def spherical_norm(tensor_list):
    return [F.normalize(vec, dim=1) for vec in spherical(tensor_list)]


def sg_product(axis1: torch.Tensor, sharpness1: torch.Tensor, axis2: torch.Tensor, sharpness2: torch.Tensor):
    axis = axis1 * sharpness1 + axis2 * sharpness2
    sharpness = torch.linalg.norm(axis, dim=1, keepdim=True)

    d = axis1 - axis2
    len2 = torch.sum(d * d, dim=1, keepdim=True)

    denom = torch.maximum(sharpness + sharpness1 + sharpness2, torch.tensor(torch.finfo(torch.float32).eps, device=sharpness.device))
    log_amplitude = -sharpness1 * sharpness2 * len2 / denom

    axis = axis / torch.maximum(sharpness, torch.tensor(torch.finfo(torch.float32).eps, device=sharpness.device))

    return axis, sharpness, log_amplitude


# [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 5]
def upper_sg_clamp_cosine_integral_over_two_pi(sharpness: torch.Tensor):
    mask = sharpness <= 0.5
    result = torch.zeros_like(sharpness)

    if mask.any():
        result[mask] = (((((((-1.0 / 362880.0) * sharpness[mask] + 1.0 / 40320.0) * sharpness[mask] - 1.0 / 5040.0) * sharpness[mask] + 1.0 / 720.0) * sharpness[mask] - 1.0 / 120.0) * sharpness[mask] + 1.0 / 24.0) * sharpness[mask] - 1.0 / 6.0) * sharpness[mask] + 0.5

    if (~mask).any():
        result[~mask] = (torch.expm1(-sharpness[~mask]) + sharpness[~mask]) / (sharpness[~mask] * sharpness[~mask])

    return result


# [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 6]
def lower_sg_clamp_cosine_integral_over_two_pi(sharpness: torch.Tensor):
    e = torch.exp(-sharpness)
    mask = sharpness <= 0.5
    result = torch.zeros_like(sharpness)

    if mask.any():
        result[mask] = e[mask] * (((((((((1.0 / 403200.0) * sharpness[mask] - 1.0 / 45360.0) * sharpness[mask] + 1.0 / 5760.0) * sharpness[mask] - 1.0 / 840.0) * sharpness[mask] + 1.0 / 144.0) * sharpness[mask] - 1.0 / 30.0) * sharpness[mask] + 1.0 / 8.0) * sharpness[mask] - 1.0 / 3.0) * sharpness[mask] + 0.5)

    if (~mask).any():
        result[~mask] = e[~mask] * (-torch.expm1(-sharpness[~mask]) - sharpness[~mask] * e[~mask]) / (sharpness[~mask] * sharpness[~mask])

    return result


# Approximate product integral of an SG and clamped cosine / pi.
# [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 7]
def sg_clamp_cosine_product_integral_over_pi(cosine: torch.Tensor, sharpness: torch.Tensor):
    A = 2.7360831611272558028247203765204
    B = 17.02129778174187535455530451145
    C = 4.0100826728510421403939290030394
    D = 15.219156263147210594866010069381
    E = 76.087896272360737270901154261082

    sqrt_term = 0.5 * ((sharpness + A) * sharpness + B) / (((sharpness + C) * sharpness + D) * sharpness + E)
    t = sharpness * torch.sqrt(sqrt_term)
    tz = t * cosine

    INV_SQRTPI = 0.56418958354775628694807945156077
    CLAMPING_THRESHOLD = 0.5 * torch.finfo(torch.float32).eps

    erfc_neg_tz = torch.erfc(-tz)
    erfc_t = torch.erfc(t)
    exp_neg_tz2 = torch.exp(-tz * tz)
    exp_safe = torch.expm1(t * t * (cosine * cosine - 1.0))

    exp_term = torch.where(t.abs() > torch.finfo(torch.float32).eps, exp_safe / t, torch.zeros_like(t))

    lerp_factor = torch.clamp(
        torch.maximum(
            0.5 * (cosine * erfc_neg_tz + erfc_t) - 0.5 * INV_SQRTPI * exp_neg_tz2 * exp_term,
            torch.tensor(CLAMPING_THRESHOLD, dtype=sharpness.dtype, device=sharpness.device)
        ),
        0.0, 1.0
    )

    lower_integral = lower_sg_clamp_cosine_integral_over_two_pi(sharpness)
    upper_integral = upper_sg_clamp_cosine_integral_over_two_pi(sharpness)

    return 2.0 * torch.lerp(lower_integral, upper_integral, lerp_factor)


def sggx(m: torch.Tensor, roughness_mat: torch.Tensor) -> torch.Tensor:
    # more safe option for det
    sign, logabsdet = torch.linalg.slogdet(roughness_mat)
    logabsdet = torch.clamp(logabsdet, min=-14.0, max=40)
    det = sign * torch.exp(logabsdet)

    roughness_mat_adj = torch.stack([
        torch.stack([roughness_mat[..., 1, 1], -roughness_mat[..., 0, 1]], dim=-1),
        torch.stack([-roughness_mat[..., 1, 0], roughness_mat[..., 0, 0]], dim=-1)
    ], dim=-2)

    m_xy = m[..., :2].unsqueeze(-2)
    term1 = (m_xy @ roughness_mat_adj @ m_xy.transpose(-2, -1)).squeeze(-1).squeeze(-1) / det
    term2 = m[..., 2] ** 2
    length2 = term1 + term2

    length2 = length2.clamp(min=1e-4)

    sqrt_det = torch.sqrt(det).clamp(min=1e-4)
    denom = sqrt_det * length2 ** 2

    result = 1.0 / (torch.pi * denom)
    return result


# Approximate the reflection lobe with an SG lobe for microfacet BRDFs.
# [Wang et al. 2009 "All-Frequency Rendering with Dynamic, Spatially-Varying Reflectance"]
def sgg_reflection_pdf(wi: torch.Tensor, m: torch.Tensor, roughness_mat: torch.Tensor) -> torch.Tensor:
    xy = wi[..., :2]
    rough_wi = torch.matmul(roughness_mat, xy.unsqueeze(-1)).squeeze(-1)
    denom = torch.sqrt(torch.sum(xy * rough_wi, dim=-1) + wi[..., 2] ** 2)
    sggx_tensor = sggx(m, roughness_mat)
    return sggx_tensor / (4.0 * denom)


# Approximate hemispherical integral for a vMF distribution (i.e. normalized SG).
# [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (Supplementary Document)" Listing. 4]
def vmf_hemispherical_integral(cosine: torch.Tensor, sharpness: torch.Tensor):
    cosine = cosine.unsqueeze(-1)
    A = 0.6517328826907056171791055021459
    B = 1.3418280033141287699294252888649
    C = 7.2216687798956709087860872386955
    steepness = sharpness * torch.sqrt((0.5 * sharpness + A) / ((sharpness + B) * sharpness + C))
    lerp_factor = torch.clamp(0.5 + 0.5 * (torch.erf(steepness * torch.clamp(cosine, -1.0, 1.0)) / torch.erf(steepness)), 0, 1)

    e = torch.exp(-sharpness)
    one_tensor = torch.tensor(1.0, device=e.device, dtype=e.dtype)
    return torch.lerp(e, one_tensor, lerp_factor) / (e + 1.0)


def luminance(color: torch.Tensor):
    with torch.no_grad():
        weights = torch.tensor([0.2126, 0.7152, 0.0722], device=color.device)
        return torch.sum(color * weights, dim=1)


def compute_jacobian(wi_tensor: torch.Tensor):
    vlen = torch.norm(wi_tensor[:, :2], dim=1)

    v = torch.where(
        vlen.unsqueeze(-1) != 0.0,
        wi_tensor[:, :2] / vlen.unsqueeze(-1),
        torch.tensor([1.0, 0.0], dtype=torch.float32, device=wi_tensor.device),
    )

    jacobian_mat_1 = torch.zeros((wi_tensor.shape[0], 2, 2), dtype=torch.float32, device=wi_tensor.device)
    jacobian_mat_1[:, 0, 0] = v[:, 0]
    jacobian_mat_1[:, 0, 1] = -v[:, 1]
    jacobian_mat_1[:, 1, 0] = v[:, 1]
    jacobian_mat_1[:, 1, 1] = v[:, 0]

    jacobian_mat_2 = torch.zeros((wi_tensor.shape[0], 2, 2), dtype=torch.float32, device=wi_tensor.device)
    jacobian_mat_2[:, 0, 0] = 0.5
    jacobian_mat_2[:, 0, 1] = 0.0
    jacobian_mat_2[:, 1, 0] = 0.0
    jacobian_mat_2[:, 1, 1] = 0.5 / wi_tensor[:, 2]

    jacobian_mat = torch.bmm(jacobian_mat_1, jacobian_mat_2)
    jj_mat = torch.bmm(jacobian_mat, jacobian_mat.transpose(1, 2))

    return jj_mat


def isotropic_ndf_filtering(si: mi.SurfaceInteraction3f):
    SIGMA2 = 0.15915494  # Variance of pixel filter kernel (1/(2pi))
    KAPPA = 0.18

    dndu = si.dn_du.torch().permute(1, 0)
    dndv = si.dn_dv.torch().permute(1, 0)

    bsdf = si.bsdf()

    has_alpha    = bsdf.has_attribute("alpha", active=True)
    has_alpha_u  = bsdf.has_attribute("alpha_u", active=True)
    has_alpha_v  = bsdf.has_attribute("alpha_v", active=True)

    alpha_mask      = has_alpha & ~has_alpha_u & ~has_alpha_v
    alpha_uv_mask   = has_alpha_u & has_alpha_v

    alpha_u = dr.zeros(mi.Float, dr.shape(si.p)[1])
    alpha_v = dr.zeros(mi.Float, dr.shape(si.p)[1])

    if dr.any(alpha_mask):
        alpha_val = bsdf.eval_attribute_1("alpha", si, active=alpha_mask)
        alpha_u = dr.select(alpha_mask, alpha_val, alpha_u)
        alpha_v = dr.select(alpha_mask, alpha_val, alpha_v)

    if dr.any(alpha_uv_mask):
        alpha_u_val = bsdf.eval_attribute_1("alpha_u", si, active=alpha_uv_mask)
        alpha_v_val = bsdf.eval_attribute_1("alpha_v", si, active=alpha_uv_mask)
        alpha_u = dr.select(alpha_uv_mask, alpha_u_val, alpha_u)
        alpha_v = dr.select(alpha_uv_mask, alpha_v_val, alpha_v)

    alpha_u = alpha_u.torch()
    alpha_v = alpha_v.torch()

    print_tensor_stats(alpha_u.unsqueeze(-1), "alphau")
    print_tensor_stats(alpha_v.unsqueeze(-1), "alphav")
    kernel_roughness2 = SIGMA2 * (torch.sum(dndu * dndu, dim=-1) + torch.sum(dndv * dndv, dim=-1))
    clamped_kernel_roughness2 = torch.clamp(kernel_roughness2, max=KAPPA).unsqueeze(-1)

    roughness = torch.stack([alpha_u, alpha_v], dim=-1)
    filtered_roughness2 = torch.clamp(roughness ** 2 + clamped_kernel_roughness2, min=0.0, max=1.0)
    return torch.sqrt(filtered_roughness2)


def compute_filtered_roughness_mat(filtered_proj_roughness_mat, tr, det):
    FLT_MAX = torch.finfo(torch.float32).max

    denom = 1.0 + tr + det
    is_finite = torch.isfinite(denom)

    det_mat = torch.zeros_like(filtered_proj_roughness_mat)
    det_mat[:, 0, 0] = det
    det_mat[:, 1, 1] = det

    mat1 = torch.clamp(filtered_proj_roughness_mat + det_mat, max=FLT_MAX) / denom.unsqueeze(-1).unsqueeze(-1)

    mat2 = torch.zeros_like(filtered_proj_roughness_mat)
    mat2[:, 0, 0] = torch.clamp(filtered_proj_roughness_mat[:, 0, 0], max=FLT_MAX) / torch.clamp(filtered_proj_roughness_mat[:, 0, 0] + 1.0, max=FLT_MAX)
    mat2[:, 1, 1] = torch.clamp(filtered_proj_roughness_mat[:, 1, 1], max=FLT_MAX) / torch.clamp(filtered_proj_roughness_mat[:, 1, 1] + 1.0, max=FLT_MAX)

    return torch.where(is_finite.unsqueeze(-1).unsqueeze(-1), mat1, mat2)


# (exp(x) - 1)/x with cancellation of rounding errors.
# [Nicholas J. Higham "Accuracy and Stability of Numerical Algorithms", Section 1.14.1, p. 19]
def expm1_over_x(x: torch.Tensor) -> torch.Tensor:
    u = torch.exp(x)
    result = torch.zeros_like(x)
    close_to_zero_mask = torch.isclose(u, torch.ones_like(u), atol=1e-6)
    result[close_to_zero_mask] = 1.0
    y = u - 1.0
    small_x_mask = ~close_to_zero_mask & (torch.abs(x) < 1.0)
    result[small_x_mask] = y[small_x_mask] / torch.log(u[small_x_mask])
    large_x_mask = ~close_to_zero_mask & ~small_x_mask
    result[large_x_mask] = y[large_x_mask] / x[large_x_mask]

    return result


def sg_integral(sharpness):
    return 4.0 * torch.pi * expm1_over_x(-2.0 * sharpness)


def orthonormal_basis(axis: torch.Tensor):
    s = torch.where(axis[:, 2] >= 0.0, 1.0, -1.0)
    c = -1.0 / (s + axis[:, 2])
    b = axis[:, 0] * axis[:, 1] * c
    b1 = torch.stack([
        1.0 + s * axis[:, 0] * axis[:, 0] * c,
        s * b,
        -s * axis[:, 0]
    ], dim=1)
    b2 = torch.stack([
        b,
        s + axis[:, 1] * axis[:, 1] * c,
        -axis[:, 1]
    ], dim=1)

    return torch.stack([b1, b2, axis], dim=1)


def sample_vmf(axis: torch.Tensor, sharpness: torch.Tensor):
    rand = torch.rand((axis.shape[0], 2), dtype=axis.dtype, device=axis.device)
    phi = 2.0 * math.pi * rand[:, 0]
    THRESHOLD = torch.finfo(torch.float32).eps / 4.0

    mask = sharpness.squeeze(-1) > THRESHOLD
    r = torch.empty_like(sharpness.squeeze(-1))

    r[mask] = torch.log1p(rand[mask, 1] * torch.expm1(-2.0 * sharpness[mask, 0])) / sharpness[mask, 0]
    r[~mask] = -2.0 * rand[~mask, 1]

    cos_theta = 1.0 + r
    sin_theta = torch.sqrt(-r * r - 2.0 * r)

    dir = torch.stack([
        torch.cos(phi) * sin_theta,
        torch.sin(phi) * sin_theta,
        cos_theta
    ], dim=1)

    frame = orthonormal_basis(axis)

    return torch.einsum('nij,nj->ni', frame, dir)


def batched_matmul(A, B, batch_size=1024):
    N = A.shape[0]
    result = []

    for i in range(0, N, batch_size):
        batch_A = A[i : i + batch_size]
        batch_B = B[i : i + batch_size]

        batch_result = torch.bmm(batch_A, batch_B)
        result.append(batch_result)

    return torch.cat(result, dim=0)


# BSDF Approximations: Diffuse as CosineLobeSG, Specular as AnisotropicSG
class cosine_lobe_sg:
    def __init__(self, direction: torch.Tensor):
        self.axis = direction
        self.sharpness = torch.full((direction.shape[0], 1), 2.123, device=direction.device, dtype=direction.dtype)
        self.amplitude = torch.full((direction.shape[0], 1), 1.17, device=direction.device, dtype=direction.dtype)


def asg_reflection_lobe(dir: torch.Tensor, normal: torch.Tensor, roughness2: torch.Tensor):
    sharpness_ndf = 1.0 / roughness2 - 1.0

    jacobian_diag_x = 2.0 * torch.sum(dir * normal, dim=1, keepdim=True)

    sharpness = sharpness_ndf / (jacobian_diag_x * jacobian_diag_x)

    axis_x = torch.nn.functional.normalize(torch.cross(dir, normal, dim=1), dim=1)
    axis_z = torch.nn.functional.normalize(dir - 2 * (torch.sum(dir * normal, dim=1, keepdim=True) * normal), dim=1)
    axis_y = torch.cross(axis_z, axis_x, dim=1)

    log_amplitude = torch.zeros_like(sharpness)

    return anisotropic_lobe_sg(axis_x, axis_y, axis_z, sharpness, log_amplitude)


class anisotropic_lobe_sg:
    def __init__(self, axis_x: torch.Tensor, axis_y: torch.Tensor, axis_z: torch.Tensor,
                 sharpness: torch.Tensor, log_amplitude: torch.Tensor):
        self.axis_x = axis_x
        self.axis_y = axis_y
        self.axis_z = axis_z
        self.sharpness = sharpness
        self.log_amplitude = log_amplitude


# vMF-vMF convolution
def A3(kappa):
    return torch.reciprocal(torch.tanh(kappa)) - torch.reciprocal(kappa)

def dA3(kappa):
    exp_kappa = torch.exp(kappa)
    exp_neg_kappa = torch.reciprocal(exp_kappa)
    csch = 2.0 * torch.reciprocal(exp_kappa - exp_neg_kappa)

    return torch.reciprocal(kappa * kappa) - csch * csch

def A3inv(y, x):
    max_iter = 25
    atol = 1e-5
    for _ in range(max_iter):
        residual = A3(x)
        residual.sub_(y)

        deriv = dA3(x)
        x.sub_(residual / deriv)

        if torch.max(torch.abs(residual)) < atol:
            break

    return x

def convolve_vmfs(kappa1, kappa2):
    return A3inv(A3(kappa1).mul_(A3(kappa2)), torch.minimum(kappa1, kappa2))


def approximate_bsdf_with_vmf(bsdf: mi.BSDFPtr, normal: torch.Tensor, view_dir: torch.Tensor, roughness2: torch.Tensor):
    num_interactions = normal.shape[0]

    bsdf_eval = bsdf.flags()

    has_diffuse = (bsdf_eval & int(mi.BSDFFlags.Diffuse.value)) != 0
    has_specular = (bsdf_eval & int(mi.BSDFFlags.Glossy.value)) != 0

    mask_diffuse = torch.from_numpy(has_diffuse.numpy()).to("cuda")
    mask_specular = torch.from_numpy(has_specular.numpy()).to("cuda")

    mask_random = torch.randint(0, 2, (num_interactions,), dtype=torch.bool, device="cuda")
    mask_final = mask_diffuse & mask_specular
    mask_diffuse = mask_diffuse & ~mask_final | (mask_final & mask_random)
    mask_specular = mask_specular & ~mask_final | (mask_final & ~mask_random)

    axis = torch.zeros((num_interactions, 3), device=normal.device)
    sharpness = torch.zeros((num_interactions, 2), device=normal.device)
    amplitude = torch.zeros((num_interactions, 1), device=normal.device)

    if mask_diffuse.any():
        lobes_diffuse = cosine_lobe_sg(normal[mask_diffuse])
        axis[mask_diffuse] = lobes_diffuse.axis
        sharpness[mask_diffuse, 0] = lobes_diffuse.sharpness.squeeze(-1)
        amplitude[mask_diffuse] = lobes_diffuse.amplitude

    if mask_specular.any():
        lobes_specular = asg_reflection_lobe(view_dir[mask_specular], normal[mask_specular], roughness2[mask_specular])
        axis[mask_specular] = lobes_specular.axis_x
        sharpness[mask_specular] = lobes_specular.sharpness
        amplitude[mask_specular] = lobes_specular.log_amplitude

    return axis, sharpness, amplitude


def compute_bsdf_weight(bsdf, bsdf_ctx, si, wo_new, active=True):
    wo = mi.Vector3f(wo_new)
    bsdf_value = bsdf.eval(bsdf_ctx, si, wo, active)
    pdf = bsdf.pdf(bsdf_ctx, si, wo, active)
    cos_theta = abs(mi.Frame3f.cos_theta(wo))
    bsdf_weight = (bsdf_value * cos_theta) / pdf

    return bsdf_weight


def print_tensor_stats(tensor, tensor_name="tensor"):
    return
    # TODO: add asserts for wrong tensors
    with torch.no_grad():
        if isinstance(tensor, torch.Tensor):
            if tensor.ndim == 2 and (tensor.shape[1] == 1 or tensor.shape[1] == 2 or tensor.shape[1] == 3):
                min_val = tensor.min(axis=0).values
                max_val = tensor.max(axis=0).values

                print(f"Tensor name: {tensor_name}")
                print(f"Min value: {min_val}")
                print(f"Max value: {max_val}")
            else:
                print("Tensor must have shape (N, 1) or (N, 3).")
        else:
            print("Input is not a valid torch tensor.")
