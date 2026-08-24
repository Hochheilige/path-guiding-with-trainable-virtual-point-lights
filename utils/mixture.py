import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import torch

from .sg_vmf import (
    spherical, spherical_norm,
    sg_product, sg_clamp_cosine_product_integral_over_pi,
    vmf_hemispherical_integral, sg_integral,
    compute_jacobian, isotropic_ndf_filtering,
    sggx, sgg_reflection_pdf,
    sample_vmf, convolve_vmfs,
    cosine_lobe_sg, asg_reflection_lobe,
    print_tensor_stats,
)
import drjit as dr


class vapl_mixture:
    def __init__(self, gaussians, vmfs, sweep_encoding):
        if not isinstance(gaussians, list):
            gaussians = [gaussians]
        if not isinstance(vmfs, list):
            vmfs = [vmfs]

        self.mean = [g[:, :3] for g in gaussians]
        self.variance = [g[:, 3] for g in gaussians]
        self.sharpness = [v[:, 0] for v in vmfs]
        # need to pass config here because axis not always 1:4 and amplitude not always 4:7
        if sweep_encoding == "spherical":
            axis = [v[:, 1:3] for v in vmfs]
            self.axis = spherical(axis)
            self.amplitude = [v[:, 3:6] for v in vmfs]
        elif sweep_encoding == "spherical-norm":
            axis = [v[:, 1:3] for v in vmfs]
            self.axis = spherical_norm(axis)
            self.amplitude = [v[:, 3:6] for v in vmfs]
        else:
            self.axis = [v[:, 1:4] for v in vmfs]
            self.amplitude = [v[:, 4:7] for v in vmfs]

        self.normalized_vapl_weights = [torch.ones(g.shape[0]) for g in gaussians]
        self.num_rays = [g.shape[0] for g in gaussians]

        self.illumination = torch.zeros(gaussians[0].shape[0], 3, device=gaussians[0].device, dtype=gaussians[0].dtype)
        self.diffuse_illumination = torch.zeros(gaussians[0].shape[0], 3, device=gaussians[0].device, dtype=gaussians[0].dtype)
        self.specular_illumination = torch.zeros(gaussians[0].shape[0], 3, device=gaussians[0].device, dtype=gaussians[0].dtype)

    def calculate_normalized_vapl_weights(self, si: mi.SurfaceInteraction3f, view_dir):
        weights = self.convolve_with_bsdf(si, view_dir)
        total_weight = torch.sum(weights)
        self.normalized_vapl_weights = weights / total_weight

    def sample_vapl(self, si: mi.SurfaceInteraction3f, view_dir):
        self.calculate_normalized_vapl_weights(si, view_dir)
        indices = torch.multinomial(self.normalized_vapl_weights, num_samples=self.num_rays, replacement=True)

        self.mean                     = self.mean[indices]
        self.variance                 = self.variance[indices]
        self.sharpness                = self.sharpness[indices]
        self.axis                     = self.axis[indices]
        self.amplitude                = self.amplitude[indices]
        self.light_lobe_axis          = self.light_lobe_axis[indices]
        self.light_lobe_sharpness     = self.light_lobe_sharpness[indices]
        self.light_lobe_log_amplitude = self.light_lobe_log_amplitude[indices]
        self.illumination             = self.illumination[indices]
        return self

    def sample_from_current_ligth_lobe_vmf(self):
        sampled_dir: torch.Tensor = sample_vmf(self.light_lobe_axis, self.light_lobe_sharpness)
        return sampled_dir.permute(1, 0)

    def sample_from_current_bsdf_light_lobe_vmf(self):
        sharpness = convolve_vmfs(self.bsdf_sharpness, self.light_lobe_sharpness)
        axis = self.bsdf_axis
        sampled_dir: torch.Tensor = sample_vmf(axis, sharpness[:, :1])
        return sampled_dir.permute(1, 0)

    def convolve(self, si: mi.SurfaceInteraction3f, view_dir: mi.Vector3f):
        for i in range(len(self.mean)):
            self.convolve_with_bsdf(i, si, view_dir)

    def convolve_with_bsdf(self, index, si: mi.SurfaceInteraction3f, view_dir: mi.Vector3f):
        SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
        eps = 1e-4

        position  = si.p
        normal    = si.n
        pos_tensor  = position.torch().permute(1, 0)
        norm_tensor = normal.torch().permute(1, 0)

        view_dir_normalize = (torch.nn.functional.normalize(view_dir.torch().permute(1, 0), p=2, dim=1, eps=1e-6))
        wo_world = -view_dir_normalize
        wi_world = view_dir_normalize
        wo = mi.Vector3f(wo_world.permute(1, 0))
        wi = si.sh_frame.to_local(mi.Vector3f(wi_world.permute(1, 0)))
        wi_tensor = wi.torch().permute(1, 0)
        wo_ts = si.sh_frame.to_local(mi.Vector3f(wo_world.permute(1, 0)))

        bsdf: mi.BSDFPtr = si.bsdf()

        light_vec = self.mean[index] - pos_tensor
        squared_distance = torch.sum(light_vec * light_vec, dim=1).unsqueeze(1)
        light_dir = light_vec * torch.rsqrt(squared_distance)

        variance = torch.maximum(self.variance[index].unsqueeze(1), squared_distance / SGLIGHT_SHARPNESS_MAX)

        emissive = self.amplitude[index] / (variance)

        light_sharpness = squared_distance / (variance)

        self.light_lobe_axis, self.light_lobe_sharpness, self.light_lobe_log_amplitude = sg_product(
            self.axis[index], self.sharpness[index].unsqueeze(1), light_dir, light_sharpness)

        # VSGL convention: raw diffuse albedo rho, view-INDEPENDENT (restores the
        # original eval_diffuse_reflectance; bsdf.eval injected a spurious cos(theta_view)).
        diffuse: mi.Spectrum = bsdf.eval_diffuse_reflectance(si)

        # Diffuse SG lighting.
        # [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 4]
        amplitude = torch.exp(self.light_lobe_log_amplitude)
        cosine = torch.clamp(torch.sum(self.light_lobe_axis * norm_tensor, dim=1), -1.0, 1.0).unsqueeze(1)

        diffuse_illumination = amplitude * sg_clamp_cosine_product_integral_over_pi(cosine, self.light_lobe_sharpness)
        diffuse_tensor: torch.Tensor = diffuse.torch().permute(1, 0)
        diffuse_illumination_result = diffuse_tensor * diffuse_illumination

        mask = wi_tensor[:, 2] == 0
        wi_tensor[mask] += eps

        jj_mat = compute_jacobian(wi_tensor)

        det_jj4 = 1.0 / (4.0 * wi_tensor[:, 2] ** 2)

        roughness = isotropic_ndf_filtering(si)
        roughness2 = roughness**2
        proj_roughness2 = roughness2 / torch.maximum(1.0 - roughness2, torch.tensor(eps, device=roughness2.device))
        roughness_max2 = torch.max(roughness2, dim=-1, keepdim=True).values

        reflect_sharpness = (1.0 - roughness_max2) / torch.maximum(2.0 * roughness_max2, torch.tensor(eps, device=roughness2.device))
        reflect_vec_tensor = mi.reflect(wo, normal).torch().permute(1, 0)
        reflect_vec = reflect_vec_tensor * reflect_sharpness

        # Glossy SG lighting.
        # [Tokuyoshi et al. 2024 "Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting", Section 5]
        prod_vec = reflect_vec + self.light_lobe_axis * self.light_lobe_sharpness
        prod_sharpness = torch.linalg.norm(prod_vec, dim=1, keepdim=True)
        prod_dir = prod_vec / prod_sharpness

        light_lobe_variance = (1.0 / self.light_lobe_sharpness).squeeze(-1)

        filtered_proj_roughness_mat = torch.zeros((proj_roughness2.shape[0], 2, 2), dtype=torch.float32, device=proj_roughness2.device)
        filtered_proj_roughness_mat[:, 0, 0] = proj_roughness2[:, 0]
        filtered_proj_roughness_mat[:, 1, 1] = proj_roughness2[:, 1]

        doubled_light_lobe_var = 2.0 * light_lobe_variance
        var_jj_mat = jj_mat.clone()
        var_jj_mat[:, 0, 0] *= doubled_light_lobe_var
        var_jj_mat[:, 0, 1] *= doubled_light_lobe_var
        var_jj_mat[:, 1, 0] *= doubled_light_lobe_var
        var_jj_mat[:, 1, 1] *= doubled_light_lobe_var

        filtered_proj_roughness_mat = filtered_proj_roughness_mat + var_jj_mat

        jj_mat_11 = jj_mat[:, 0, 0]
        jj_mat_22 = jj_mat[:, 1, 1]
        det = (proj_roughness2[:, 0] * proj_roughness2[:, 1]) \
            + 2.0 * light_lobe_variance * (proj_roughness2[:, 0] * jj_mat_11 + proj_roughness2[:, 1] * jj_mat_22) \
            + light_lobe_variance * light_lobe_variance * det_jj4

        tr = filtered_proj_roughness_mat[:, 0, 0] + filtered_proj_roughness_mat[:, 1, 1]

        is_finite = torch.isfinite(1.0 + tr + det)
        flt_max = torch.tensor(torch.finfo(torch.float32).max, device=det.device)
        filtered_roughness_mat = torch.zeros_like(filtered_proj_roughness_mat)

        denom = 1.0 + tr + det

        filtered_roughness_mat[:, 0, 0] = torch.minimum(
            filtered_proj_roughness_mat[:, 0, 0] + det, flt_max
        ) / denom
        filtered_roughness_mat[:, 1, 1] = torch.minimum(
            filtered_proj_roughness_mat[:, 1, 1] + det, flt_max
        ) / denom

        filtered_roughness_mat[~is_finite, 0, 0] = torch.minimum(
            filtered_proj_roughness_mat[~is_finite, 0, 0], flt_max
        ) / torch.minimum(filtered_proj_roughness_mat[~is_finite, 0, 0] + 1.0, flt_max)

        filtered_roughness_mat[~is_finite, 1, 1] = torch.minimum(
            filtered_proj_roughness_mat[~is_finite, 1, 1], flt_max
        ) / torch.minimum(filtered_proj_roughness_mat[~is_finite, 1, 1] + 1.0, flt_max)

        visibility = vmf_hemispherical_integral(torch.sum(prod_dir * norm_tensor, dim=1), prod_sharpness)
        print_tensor_stats(visibility, "visibility")

        wo_tensor = wo_ts.torch().permute(1, 0)
        light_lobe_axis_tf = si.sh_frame.to_local(mi.Vector3f(self.light_lobe_axis.permute(1, 0)))
        half_vec_unnormalize = wo_tensor + light_lobe_axis_tf.torch().permute(1, 0)
        half_vec = torch.nn.functional.normalize(half_vec_unnormalize, p=2, dim=1, eps=1e-6)

        lobe = sgg_reflection_pdf(wo_tensor, half_vec, filtered_roughness_mat).unsqueeze(1)
        print_tensor_stats(lobe, "lobe")

        sg_int = sg_integral(self.light_lobe_sharpness)
        print_tensor_stats(sg_int, "sg int")

        has_glossy = mi.has_flag(bsdf.flags(), mi.BSDFFlags.Glossy)
        has_specular_reflectance = bsdf.has_attribute("specular_reflectance", active=True)
        specular_reflectance = dr.select(
            has_glossy,
            dr.select(has_specular_reflectance,
                       bsdf.eval_attribute_3("specular_reflectance", si, active=has_specular_reflectance),
                       mi.Spectrum(1.0)),
            mi.Spectrum(0.0)
        )
        specular_tensor = specular_reflectance.torch().permute(1, 0)

        specular_illumination = amplitude * visibility * lobe * sg_int
        specular_illumination_result = specular_tensor * specular_illumination
        print_tensor_stats(specular_illumination_result, "specular result")
        result = emissive * (diffuse_illumination_result + specular_illumination_result)
        print_tensor_stats(result, "result")

        self.diffuse_illumination = self.diffuse_illumination + diffuse_illumination_result
        self.specular_illumination = self.specular_illumination + specular_illumination_result
        self.illumination = self.illumination + result
