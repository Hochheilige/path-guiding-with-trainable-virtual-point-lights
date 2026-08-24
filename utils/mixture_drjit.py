import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from .sg_vmf_drjit import (
    sg_product, sg_clamp_cosine_product_integral_over_pi,
    vmf_hemispherical_integral, sg_integral,
    compute_jacobian, isotropic_ndf_filtering,
    sggx, sgg_reflection_pdf,
    sample_vmf_with_samples, vmf_pdf,
)


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

        self.illumination = mi.Color3f(0.0)
        self.diffuse_illumination = mi.Color3f(0.0)
        self.specular_illumination = mi.Color3f(0.0)

        self.light_lobe_axes = []
        self.light_lobe_sharpnesses = []
        self.vapl_weights = []

    def compute_vapl_weights(self, si, view_dir):
        """Compute normalized per-level weights using the full BSDF convolution.

        Mirrors calculate_normalized_vapl_weights() from the PyTorch class:
        run the full illumination convolution for each level independently,
        use the luminance of each level's contribution as its sampling weight.
        Stores result in self.vapl_weights (list of mi.Float, one per level).
        """
        saved = (self.illumination, self.diffuse_illumination, self.specular_illumination)

        weights = []
        for i in range(len(self.mean)):
            self.illumination          = mi.Color3f(0.0)
            self.diffuse_illumination  = mi.Color3f(0.0)
            self.specular_illumination = mi.Color3f(0.0)

            self._convolve_single(i, si, view_dir)

            lum = dr.maximum(
                0.2126 * self.illumination.x
                + 0.7152 * self.illumination.y
                + 0.0722 * self.illumination.z,
                mi.Float(0.0)
            )
            weights.append(lum)

        self.illumination, self.diffuse_illumination, self.specular_illumination = saved

        total = dr.maximum(sum(weights), mi.Float(1e-6))
        self.vapl_weights = [w / total for w in weights]

    def compute_light_lobes(self, si):
        """Compute the IS sampling lobe for every VAPL level.

        Uses the light_lobe_axis from sg_product — the axis of the combined
        VAPL×geometry lobe.  This is the most principled IS direction: it blends
        the trained vMF axis with the geometric direction toward the VAPL mean,
        weighted by their sharpnesses, giving the direction the VAPL actually
        contributes light from at this surface point.

        Stores results in self.light_lobe_axes and self.light_lobe_sharpnesses.
        """
        SGLIGHT_SHARPNESS_MAX = float.fromhex("0x1.0p41")
        eps = 1e-4

        self.light_lobe_axes = []
        self.light_lobe_sharpnesses = []

        for i in range(len(self.mean)):
            light_vec        = self.mean[i] - si.p
            squared_distance = dr.dot(light_vec, light_vec)
            light_dir        = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))

            variance        = dr.maximum(self.variance[i], squared_distance / SGLIGHT_SHARPNESS_MAX)
            light_sharpness = squared_distance / dr.maximum(variance, eps)

            lobe_axis, lobe_sharpness, _ = sg_product(
                self.axis[i], self.sharpness[i], light_dir, light_sharpness
            )

            self.light_lobe_axes.append(lobe_axis)
            self.light_lobe_sharpnesses.append(lobe_sharpness)

    def sample_from_light_lobes(self, sampler):
        """Sample a direction from the mixture of all level light lobes.

        Levels are weighted by the luminance of their amplitude — the same weighting
        concept as calculate_normalized_vapl_weights() in the PyTorch class, but using
        amplitude luminance as a cheap proxy instead of the full BSDF convolution.

        Returns (wo: mi.Vector3f, pdf: mi.Float).
        Call compute_light_lobes() first.
        """
        N   = len(self.light_lobe_axes)
        eps = 1e-6

        # Use full-convolution weights if available (compute_vapl_weights was called),
        # otherwise fall back to amplitude luminance
        if self.vapl_weights:
            norm_w = self.vapl_weights
        else:
            weights = [0.2126 * self.amplitude[i].x
                     + 0.7152 * self.amplitude[i].y
                     + 0.0722 * self.amplitude[i].z
                       for i in range(N)]
            total_w = dr.maximum(sum(weights), eps)
            norm_w  = [w / total_w for w in weights]

        # CDF thresholds for level selection
        cum = [norm_w[0]]
        for i in range(1, N - 1):
            cum.append(cum[-1] + norm_w[i])

        # Sample a direction from every level (DrJIT evaluates all, selects one)
        dirs = [
            sample_vmf_with_samples(
                self.light_lobe_axes[i], self.light_lobe_sharpnesses[i],
                sampler.next_2d()
            )
            for i in range(N)
        ]

        # Select level using CDF and a single uniform sample
        rand_level = sampler.next_1d()
        wo          = dirs[-1]
        chosen_axis = self.light_lobe_axes[-1]
        chosen_sh   = self.light_lobe_sharpnesses[-1]
        for i in range(N - 2, -1, -1):
            sel = rand_level < cum[i]
            wo          = dr.select(sel, dirs[i],                   wo)
            chosen_axis = dr.select(sel, self.light_lobe_axes[i],   chosen_axis)
            chosen_sh   = dr.select(sel, self.light_lobe_sharpnesses[i], chosen_sh)

        # PDF of the selected level only — mirrors the PyTorch sample_vapl approach
        pdf = dr.maximum(vmf_pdf(wo, chosen_axis, chosen_sh), eps)

        return wo, pdf

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

        # VSGL convention: raw diffuse albedo rho, view-INDEPENDENT.
        # (bsdf.eval(ctx, si, wo) returns f * cos(theta_wo) per Mitsuba's transport
        # convention — using it here injected a spurious cos(theta_view) that made
        # the cache view-dependent and self-training divergent)
        diffuse = bsdf.eval_diffuse_reflectance(si)

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

        dr.eval(position, normal, diffuse, wo_local, wi_local,
                jj00, jj01, jj10, jj11, det_jj4,
                proj_roughness2_u, proj_roughness2_v,
                reflect_vec_scaled, specular_reflectance)

        # --- Step 2: Tile surface quantities N times (R -> N*R) ---
        tile_idx = dr.arange(mi.UInt32, total) % R

        def tile_f(arr):
            return dr.gather(mi.Float, arr, tile_idx)

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

        sh_s = si.sh_frame.s
        sh_t = si.sh_frame.t
        sh_n = si.sh_frame.n
        dr.eval(sh_s, sh_t, sh_n)
        t_sh_s = mi.Vector3f(tile_f(sh_s.x), tile_f(sh_s.y), tile_f(sh_s.z))
        t_sh_t = mi.Vector3f(tile_f(sh_t.x), tile_f(sh_t.y), tile_f(sh_t.z))
        t_sh_n = mi.Vector3f(tile_f(sh_n.x), tile_f(sh_n.y), tile_f(sh_n.z))

        # --- Step 3: Concatenate all VAPL parameters (N*R) ---
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

        light_vec = mean - position
        squared_distance = dr.dot(light_vec, light_vec)
        light_dir = light_vec * dr.rsqrt(dr.maximum(squared_distance, eps))

        variance = dr.maximum(variance, squared_distance / SGLIGHT_SHARPNESS_MAX)

        inv_variance = 1.0 / dr.maximum(variance, eps)
        emissive = amplitude * inv_variance

        light_sharpness = squared_distance / dr.maximum(variance, eps)

        light_lobe_axis, light_lobe_sharpness, light_lobe_log_amplitude = sg_product(
            axis, sharpness, light_dir, light_sharpness)

        # --- Diffuse SG lighting ---
        # VSGL convention: raw diffuse albedo rho, view-INDEPENDENT
        diffuse = bsdf.eval_diffuse_reflectance(si)

        amp = dr.exp(light_lobe_log_amplitude)
        cosine = dr.clamp(dr.dot(light_lobe_axis, normal), -1.0, 1.0)

        diffuse_integral = sg_clamp_cosine_product_integral_over_pi(cosine, light_lobe_sharpness)
        diffuse_illumination = diffuse * (amp * diffuse_integral)

        # --- Glossy SG lighting ---
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

        prod_vec = reflect_vec_scaled + light_lobe_axis * light_lobe_sharpness
        prod_sharpness = dr.norm(prod_vec)
        prod_dir = prod_vec / dr.maximum(prod_sharpness, eps)

        light_lobe_variance = 1.0 / dr.maximum(light_lobe_sharpness, eps)

        doubled_var = 2.0 * light_lobe_variance
        fr00 = proj_roughness2_u + doubled_var * jj00
        fr01 = doubled_var * jj01
        fr10 = doubled_var * jj10
        fr11 = proj_roughness2_v + doubled_var * jj11

        det = (proj_roughness2_u * proj_roughness2_v
               + 2.0 * light_lobe_variance * (proj_roughness2_u * jj00 + proj_roughness2_v * jj11)
               + light_lobe_variance * light_lobe_variance * det_jj4)

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

        visibility = vmf_hemispherical_integral(dr.dot(prod_dir, normal), prod_sharpness)

        wo_local_x = wo_local.x
        wo_local_y = wo_local.y
        wo_local_z = wo_local.z

        light_lobe_axis_local = si.sh_frame.to_local(light_lobe_axis)
        half_vec_unnorm = mi.Vector3f(wo_local_x, wo_local_y, wo_local_z) + light_lobe_axis_local
        half_vec = dr.normalize(half_vec_unnorm)

        lobe = sgg_reflection_pdf(wo_local_x, wo_local_y, wo_local_z,
                                  half_vec.x, half_vec.y, half_vec.z,
                                  filtered_r00, filtered_r01, filtered_r10, filtered_r11)

        sg_int = sg_integral(light_lobe_sharpness)

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

        self.diffuse_illumination = self.diffuse_illumination + diffuse_illumination
        self.specular_illumination = self.specular_illumination + specular_illumination
        self.illumination = self.illumination + result
