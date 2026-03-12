import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

import torch
torch.autograd.set_detect_anomaly(True)

from utils.mixture import vapl_mixture
from utils.mixture_drjit import vapl_mixture_drjit

# Base idea of integrator is taken from:
# https://github.com/krafton-ai/neural-radiosity-tutorial-mitsuba3/blob/main/neural_radiosity.ipynb

@dr.syntax
def first_non_specular_or_null_si(scene, si, sampler, β):
    """Find the first non-specular or null surface interaction."""
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()
        depth = mi.UInt32(0)
        bsdf = si.bsdf()

        null_face = ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.BackSide) & (si.wi.z < 0)
        active = si.is_valid() & ~null_face  # non-null surface
        active &= ~mi.has_flag(si.bsdf().flags(), mi.BSDFFlags.Smooth)  # Delta surface

        max_depth = 6

        while active & (depth < max_depth):
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=depth == 0
            )
            bsdf = si.bsdf(ray)

            β *= bsdf_weight
            depth[si.is_valid()] += 1

            null_face &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.BackSide) & (si.wi.z < 0)
            active &= si.is_valid() & ~null_face
            active &= ~mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

    return si, β, null_face


@dr.syntax
def _nrc_suffix_target(scene, si2, sampler, beta2, model, ray2, n_suffix):
    """NRC suffix-path training target with self-training (Mueller et al. 2021, Section 4).

    Traces n_suffix bounces from x_2 with full MIS+NEE, then queries the cache at
    the terminal vertex x_k with a stop-gradient.  This is the paper's core training
    mechanism: the cache bootstraps itself by using its own current estimate at the
    path's tail, letting short suffix paths (n_suffix=2) represent infinite-bounce
    transport after enough training iterations.

    Called inside dr.suspend_grad() → model(x_k) is automatically detached
    (no AD edge through the terminal cache query).
    """
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()
        L        = mi.Spectrum(0)
        beta     = mi.Spectrum(beta2)
        si       = si2
        ray      = ray2
        depth    = mi.UInt32(0)
        active   = si.is_valid()

        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        while active & (depth < mi.UInt32(n_suffix)):
            bsdf = si.bsdf()

            # MIS-weighted emission at current vertex
            ds_prev = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            em_pdf  = scene.pdf_emitter_direction(prev_si, ds_prev, ~prev_bsdf_delta & active)
            mis_b   = dr.select(prev_bsdf_delta, mi.Float(1.0), mis_weight(prev_bsdf_pdf, em_pdf))
            L += beta * mis_b * si.emitter(scene).eval(si)

            # NEE
            active_em            = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            ds_em, em_weight     = scene.sample_emitter_direction(si, sampler.next_2d(), True, active_em)
            active_em           &= ds_em.pdf != 0.0
            bsdf_val, bsdf_pdf_e = bsdf.eval_pdf(bsdf_ctx, si, si.to_local(ds_em.d), active_em)
            mis_em               = dr.select(ds_em.delta, mi.Float(1.0), mis_weight(ds_em.pdf, bsdf_pdf_e))
            L += dr.select(active_em, beta * mis_em * bsdf_val * em_weight, mi.Spectrum(0))

            # BSDF sample → next vertex
            bsdf_sample, bsdf_w = bsdf.sample(bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active)
            beta *= bsdf_w

            prev_si         = dr.detach(si, True)
            prev_bsdf_pdf   = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            ray    = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si     = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=False)
            active &= si.is_valid()
            depth  += 1

        # Terminal self-training cache query — detached (no grad through here).
        # model() is called inside suspend_grad so the scatter mlp_coeffs←opt['nrc']
        # creates no AD edge; the MLP output has no gradient.  This is the
        # "stop-gradient on the target" required to avoid gradient loops.
        term = mi.Color3f(model(si, ray))
        term = mi.Color3f(dr.maximum(term.x, mi.Float(0.0)),
                          dr.maximum(term.y, mi.Float(0.0)),
                          dr.maximum(term.z, mi.Float(0.0)))
        L += dr.select(active, beta * term, mi.Spectrum(0))
        return L


@dr.syntax
def path_trace_from_si(scene, si, sampler, beta, max_depth=4, rr_depth=2, indirect_only=False):
    """Multi-bounce path tracer starting from a given surface interaction.

    Implements proper two-sample MIS:
    - Emitter direction sampling (NEE) weighted against BSDF PDF
    - BSDF-sampled emitter hits weighted against emitter PDF

    indirect_only=True: skips Le and NEE at depth=0, so only light that
    has bounced at least once (indirect illumination) is returned.
    """
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()
        L        = mi.Spectrum(0)
        depth    = mi.UInt32(0)

        # Track the previous BSDF sample to MIS-weight emitter hits correctly.
        # prev_bsdf_delta=True at depth 0 so the first emitter hit (Le) counts with weight 1.
        prev_si         = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf   = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        active = si.is_valid()

        while active & (depth < mi.UInt32(max_depth)):
            bsdf = si.bsdf()

            # --- Emission at current hit, MIS-weighted against emitter sampling ---
            ds_prev         = mi.DirectionSample3f(scene, si=si, ref=prev_si)
            em_pdf_for_prev = scene.pdf_emitter_direction(
                prev_si, ds_prev, ~prev_bsdf_delta & active
            )
            mis_bsdf_w = dr.select(
                prev_bsdf_delta,
                mi.Float(1.0),
                mis_weight(prev_bsdf_pdf, em_pdf_for_prev)
            )
            Le_contrib = beta * mis_bsdf_w * si.emitter(scene).eval(si)
            # indirect_only: skip Le at depth=0 (direct emission / first-hit emitter via BSDF)
            L += dr.select(mi.Bool(not indirect_only) | (depth > mi.UInt32(0)),
                           Le_contrib, mi.Spectrum(0))

            # --- NEE: sample an emitter direction, MIS-weighted against BSDF ---
            active_em = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
            # indirect_only: skip direct NEE at depth=0
            active_em &= mi.Bool(not indirect_only) | (depth > mi.UInt32(0))
            ds_em, em_weight = scene.sample_emitter_direction(
                si, sampler.next_2d(), True, active_em
            )
            active_em &= ds_em.pdf != 0.0
            wo_em = si.to_local(ds_em.d)
            bsdf_val, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_em, active_em)
            mis_em = dr.select(ds_em.delta, mi.Float(1.0), mis_weight(ds_em.pdf, bsdf_pdf_em))
            L += dr.select(active_em, beta * mis_em * bsdf_val * em_weight, mi.Spectrum(0))

            # --- BSDF sampling: choose next direction ---
            bsdf_sample, bsdf_weight = bsdf.sample(
                bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active
            )
            beta *= bsdf_weight

            # --- Russian roulette ---
            rr_active = active & (depth >= mi.UInt32(rr_depth))
            q         = dr.minimum(dr.max(beta), mi.Float(0.95))
            survive   = sampler.next_1d() < q
            active   &= ~rr_active | survive
            beta      = dr.select(rr_active & survive, beta / q, beta)

            # --- Advance to next intersection ---
            prev_si         = dr.detach(si, True)
            prev_bsdf_pdf   = bsdf_sample.pdf
            prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

            ray_next = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si       = scene.ray_intersect(ray_next, ray_flags=mi.RayFlags.All, coherent=False)
            active  &= si.is_valid()
            depth   += 1

        return L


def compute_direct_at_si(scene, si, sampler, beta):
    """Direct illumination at si: emitter emission + one NEE shadow ray (MIS).

    Always computed without gradients — safe to call inside dr.resume_grad().
    Used by the NRC training path so the network only learns indirect light,
    while direct is handled explicitly here and added to the rendered output.
    """
    with dr.suspend_grad():
        bsdf_ctx  = mi.BSDFContext()
        bsdf      = si.bsdf()
        active    = si.is_valid()
        L         = mi.Spectrum(0)

        # Le: emission at si (non-zero when si is on an area light)
        L += dr.select(active, beta * si.emitter(scene).eval(si), mi.Spectrum(0))

        # NEE: one shadow ray to an emitter, MIS-weighted against BSDF PDF
        active_em            = active & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        ds_em, em_weight     = scene.sample_emitter_direction(
                                   si, sampler.next_2d(), True, active_em)
        active_em           &= ds_em.pdf != 0.0
        wo_em                = si.to_local(ds_em.d)
        bsdf_val, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo_em, active_em)
        mis_em               = dr.select(ds_em.delta, mi.Float(1.0),
                                         mis_weight(ds_em.pdf, bsdf_pdf_em))
        L += dr.select(active_em, beta * mis_em * bsdf_val * em_weight, mi.Spectrum(0))

    return L


# =============================================================================
# Loss functions — unified (PyTorch + DrJIT)
# =============================================================================

class LossFn:
    """Unified loss for both PyTorch and DrJIT backends.

    Initialise once with a name from config.loss, then call:
      loss_fn.torch(pred, target)   — PyTorch tensors  (regular / mlp grids)
      loss_fn.drjit(pred, target)   — DrJIT Color3f    (drjit / nrc-drjit grids)

    Available names: "relative_l2", "relative_l2_luminance",
                     "mse", "log_relative", "smape"
    """

    NAMES = {"relative_l2", "relative_l2_luminance", "mse", "log_relative", "smape"}

    def __init__(self, name: str = "relative_l2"):
        if name not in self.NAMES:
            raise ValueError(f"Unknown loss '{name}'. Choose from: {sorted(self.NAMES)}")
        self.name = name

    # ------------------------------------------------------------------
    # Public dispatch
    # ------------------------------------------------------------------

    def torch(self, pred, target):
        return self._TORCH[self.name](pred, target)

    def drjit(self, pred, target):
        return self._DRJIT[self.name](pred, target)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_filter(loss):
        valid = torch.isfinite(loss)
        if not torch.all(valid):
            loss = loss[valid]
        if loss.numel() == 0:
            return torch.tensor(0.0, device=loss.device, requires_grad=True)
        return loss.mean()

    @staticmethod
    def _drjit_filter(loss_per_ray):
        valid = dr.isfinite(loss_per_ray)
        loss_per_ray = dr.select(valid, loss_per_ray, mi.Float(0.0))
        n = mi.Float(dr.width(loss_per_ray))
        return dr.sum(loss_per_ray) / dr.maximum(n, mi.Float(1.0))

    # ------------------------------------------------------------------
    # PyTorch implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _torch_relative_l2(pred, target):
        """Relative L2 matching tiny-cuda-nn (NRC paper authors):
        each channel divided by its own squared prediction value.
        loss = Σ_c (pred_c - target_c)² / (pred_c² + ε)
        Source: tiny-cuda-nn/include/tiny-cuda-nn/losses/relative_l2.h"""
        eps = 1e-2
        denom = pred.detach() ** 2 + eps   # per-channel, shape (N, C)
        return LossFn._torch_filter((pred - target) ** 2 / denom)

    @staticmethod
    def _torch_relative_l2_luminance(pred, target):
        """Relative L2 with luminance denominator: denom = lum(pred_stop)² + ε."""
        eps = 1e-2
        lum = (pred.detach() * torch.tensor([0.2126, 0.7152, 0.0722], device=pred.device)).sum(dim=1, keepdim=True)
        denom = lum ** 2 + eps
        return LossFn._torch_filter((pred - target) ** 2 / denom)

    @staticmethod
    def _torch_mse(pred, target):
        return LossFn._torch_filter((pred - target) ** 2)

    @staticmethod
    def _torch_log_relative(pred, target):
        eps = 1e-4
        log_pred   = torch.log1p(pred.clamp(min=0) / eps)
        log_target = torch.log1p(target.clamp(min=0) / eps)
        return LossFn._torch_filter((log_pred - log_target) ** 2)

    @staticmethod
    def _torch_smape(pred, target):
        eps = 1e-2
        diff  = torch.abs(pred - target)
        denom = torch.abs(pred.detach()) + torch.abs(target) + eps
        return LossFn._torch_filter(diff / denom)

    # ------------------------------------------------------------------
    # DrJIT implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _drjit_relative_l2(pred, target):
        """Relative L2 matching tiny-cuda-nn (NRC paper authors):
        each channel divided by its own squared prediction value.
        loss = Σ_c (pred_c - target_c)² / (pred_c² + ε)
        Source: tiny-cuda-nn/include/tiny-cuda-nn/losses/relative_l2.h"""
        eps = 1e-2
        pred_d = dr.detach(pred)
        diff   = pred - target
        loss   = (diff.x * diff.x / (pred_d.x * pred_d.x + eps)
                + diff.y * diff.y / (pred_d.y * pred_d.y + eps)
                + diff.z * diff.z / (pred_d.z * pred_d.z + eps))
        return LossFn._drjit_filter(loss)

    @staticmethod
    def _drjit_relative_l2_luminance(pred, target):
        """Relative L2 with luminance denominator: denom = lum(pred_stop)² + ε."""
        eps = 1e-2
        pred_d = dr.detach(pred)
        lum    = 0.2126 * pred_d.x + 0.7152 * pred_d.y + 0.0722 * pred_d.z
        denom  = lum * lum + eps
        return LossFn._drjit_filter(dr.squared_norm(pred - target) / denom)

    @staticmethod
    def _drjit_mse(pred, target):
        return LossFn._drjit_filter(dr.squared_norm(pred - target))

    @staticmethod
    def _drjit_log_relative(pred, target):
        eps        = 1e-4
        log_pred   = dr.log(1.0 + dr.maximum(pred,   mi.Spectrum(0)) / eps)
        log_target = dr.log(1.0 + dr.maximum(target, mi.Spectrum(0)) / eps)
        return LossFn._drjit_filter(dr.squared_norm(log_pred - log_target))

    @staticmethod
    def _drjit_smape(pred, target):
        eps    = 1e-2
        pred_d = dr.detach(pred)
        diff   = dr.abs(pred - target)
        loss   = (diff.x / (dr.abs(pred_d.x) + dr.abs(target.x) + eps)
                + diff.y / (dr.abs(pred_d.y) + dr.abs(target.y) + eps)
                + diff.z / (dr.abs(pred_d.z) + dr.abs(target.z) + eps))
        return LossFn._drjit_filter(loss)


LossFn._TORCH = {
    "relative_l2":           LossFn._torch_relative_l2,
    "relative_l2_luminance": LossFn._torch_relative_l2_luminance,
    "mse":                   LossFn._torch_mse,
    "log_relative":          LossFn._torch_log_relative,
    "smape":                 LossFn._torch_smape,
}
LossFn._DRJIT = {
    "relative_l2":           LossFn._drjit_relative_l2,
    "relative_l2_luminance": LossFn._drjit_relative_l2_luminance,
    "mse":                   LossFn._drjit_mse,
    "log_relative":          LossFn._drjit_log_relative,
    "smape":                 LossFn._drjit_smape,
}


class RHSIntegrator(ADIntegrator):
    def __init__(self, model, train, loss_name: str = "relative_l2",
                 sweep_encoding=None, indirect_only=False, nrc_depth=2, props=mi.Properties()):
        super().__init__(props)
        self.train = train
        self.model = model
        self.losses = []
        self.loss_fn = LossFn(loss_name)
        self.sweep_encoding = sweep_encoding
        self.indirect_only = indirect_only
        self.depth = 1
        self.path_trace = False
        self.cache_only = False
        self.nrc_depth = nrc_depth

    def set_train(self, train):
        self.train = train

    def set_path_trace(self, enabled):
        self.path_trace = enabled

    def set_cache_only(self, enabled):
        """Inference mode that returns only bsdf_w * cache(x2), no direct(x1).
        Useful for visualising what the grid has learned at x2 in isolation."""
        self.cache_only = enabled

    def set_depth(self, depth):
        self.depth = depth

    def set_config(self, conf):
        self.sweep_encoding = conf

    def sample_training(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, depth: mi.UInt32):
        ray = mi.Ray3f(dr.detach(ray))
        β = mi.Spectrum(1)

        si = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=(depth==0)
        )

        # Skip past delta/null surfaces
        si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)

        si.compute_uv_partials(ray)

        # Compute ground truth: full multi-bounce path trace from the current si
        gt_light = path_trace_from_si(scene, si, sampler, mi.Spectrum(β),
                                      max_depth=self.depth, rr_depth=2)

        # Compute VAPL prediction at the same si
        gaussians, vmfs = self.model(si)
        mixture = vapl_mixture(gaussians, vmfs, self.sweep_encoding)
        mixture.convolve(si, ray.d)

        Le = si.emitter(scene).eval(si)
        Le_torch = Le.torch().permute(1, 0)
        vapl_l = mixture.illumination + Le_torch

        return vapl_l, gt_light, si

    def sample(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL,
               δaovs,
               state_in,
               active):

        if self.path_trace:
            ray_d  = mi.Ray3f(dr.detach(ray))
            beta   = mi.Spectrum(1)
            si_raw = scene.ray_intersect(ray_d, ray_flags=mi.RayFlags.All, coherent=(depth == 0))
            si, beta, _ = first_non_specular_or_null_si(scene, si_raw, sampler, beta)
            L = path_trace_from_si(scene, si, sampler, beta,
                                   max_depth=self.depth, rr_depth=2,
                                   indirect_only=self.indirect_only)
            return L, si.is_valid(), [], mi.Spectrum(0)

        if self.train:
            self.model.set_current_epoch(self.epoch)

            if getattr(self.model, '_is_drjit', False):
                # DrJIT-native training path.
                # ADIntegrator.render() wraps sample() in dr.suspend_grad(),
                # which suppresses DrJIT AD. We must resume it explicitly.
                with dr.resume_grad():
                    # Shared: find first non-specular hit (already suspends grad internally)
                    ray_d  = mi.Ray3f(dr.detach(ray))
                    beta   = mi.Spectrum(1)
                    si_raw = scene.ray_intersect(ray_d, ray_flags=mi.RayFlags.All, coherent=(depth == 0))
                    si, beta, _ = first_non_specular_or_null_si(scene, si_raw, sampler, beta)
                    si.compute_uv_partials(ray_d)

                    # Model prediction at the same si (gradients flow through model)
                    if getattr(self.model, '_is_nrc', False):
                        # NRC paper (Section 4): train the cache at x_2 (the BSDF-sampled
                        # vertex from x_1), NOT at x_1 (the camera-visible vertex).
                        # Inference queries the cache at x_2, so training must do the same
                        # to avoid a surface-distribution mismatch that causes color artifacts
                        # in complex scenes (country kitchen, veach mis).
                        with dr.suspend_grad():
                            direct_x1 = compute_direct_at_si(scene, si, sampler, beta)

                            # BSDF sample: x_1 → x_2
                            bsdf_sample_tr, bsdf_w_tr = si.bsdf().sample(
                                mi.BSDFContext(), si,
                                sampler.next_1d(), sampler.next_2d(), si.is_valid()
                            )
                            beta2_tr = beta * bsdf_w_tr
                            ray2_tr  = si.spawn_ray(si.to_world(bsdf_sample_tr.wo))
                            si2_tr   = scene.ray_intersect(ray2_tr, ray_flags=mi.RayFlags.All, coherent=False)
                            dr.eval(si2_tr)

                            # Training target at x_2: short suffix path terminating into
                            # a detached cache query (self-training bootstrap).
                            # With n_suffix=0 this falls back to a plain path trace from x_2.
                            n_suffix = self.nrc_depth
                            if n_suffix > 0:
                                gt_light = _nrc_suffix_target(
                                    scene, si2_tr, sampler, beta2_tr,
                                    self.model, ray2_tr, n_suffix
                                )
                            else:
                                gt_light = path_trace_from_si(
                                    scene, si2_tr, sampler, beta2_tr,
                                    max_depth=self.depth, rr_depth=2,
                                    indirect_only=False
                                )

                        # Primary cache prediction at x_2 — gradients flow here.
                        # The weight scatter mlp_coeffs←opt['nrc'] happens again here
                        # (inside resume_grad) creating the correct AD edge.
                        nrc_pred = dr.maximum(beta2_tr * self.model(si2_tr, ray2_tr), mi.Color3f(0.0))
                        loss = self.loss_fn.drjit(nrc_pred, gt_light)
                        dr.backward(self.model.scaler.scale(loss))
                        self.model.optimizer.step()
                        self.losses.append(float(loss[0]))

                        # Output: direct at x_1 + cache contribution from x_2
                        vapl_light = direct_x1 + dr.select(si2_tr.is_valid(), nrc_pred, mi.Spectrum(0))
                        return vapl_light, si.is_valid(), [], mi.Spectrum(0)
                    else:
                        # GT: path trace from the first diffuse hit (no gradients)
                        with dr.suspend_grad():
                            gt_light = path_trace_from_si(scene, si, sampler, beta,
                                                          max_depth=self.depth, rr_depth=2,
                                                          indirect_only=self.indirect_only)
                        # VAPL prediction
                        gaussians_list, vmfs_list = self.model(si)
                        for mean, var in gaussians_list:
                            dr.eval(mean, var)
                        for sh, ax, amp in vmfs_list:
                            dr.eval(sh, ax, amp)
                        mixture = vapl_mixture_drjit(gaussians_list, vmfs_list, self.sweep_encoding)
                        mixture.convolve(si, ray_d.d)
                        Le            = mi.Spectrum(0) if self.indirect_only else si.emitter(scene).eval(si)
                        vapl_light    = dr.maximum(beta * (mixture.illumination + Le), mi.Color3f(0.0))
                        pred_for_loss = vapl_light

                    loss = self.loss_fn.drjit(pred_for_loss, gt_light)
                    dr.backward(self.model.scaler.scale(loss))
                    self.model.optimizer.step()

                self.losses.append(float(loss[0]))

                return vapl_light, si.is_valid(), [], mi.Spectrum(0)
            else:
                # PyTorch training path (original)
                vapl_light, gt_light, si = self.sample_training(scene, sampler, ray, depth)
                GT_Light = torch.from_numpy(gt_light.numpy()).to("cuda").T

                loss : torch.Tensor = self.loss_fn.torch(vapl_light, GT_Light)
                self.losses.append(loss.detach().cpu())
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

                torch.cuda.empty_cache()
                return vapl_light.clamp(min=0).permute(1, 0), si.is_valid(), [], mi.Spectrum(0)
        else:
            # Inference: NRC paper Fig 2 — 1-bounce path with NEE at x1,
            # then terminate into the cache at x2.
            # L = Le(x1) + NEE(x1) + bsdf_weight(x1→x2) * cache(x2)
            ray_d  = mi.Ray3f(dr.detach(ray))
            beta   = mi.Spectrum(1)
            si_raw = scene.ray_intersect(ray_d, ray_flags=mi.RayFlags.All, coherent=(depth == 0))
            si, beta, _ = first_non_specular_or_null_si(scene, si_raw, sampler, beta)

            # Direct illumination at x1: Le + one NEE shadow ray
            L = mi.Spectrum(0) if self.cache_only else compute_direct_at_si(scene, si, sampler, beta)
            dr.eval(L)

            # BSDF sample at x1 → trace to x2
            bsdf_sample, bsdf_weight = si.bsdf().sample(
                mi.BSDFContext(), si, sampler.next_1d(), sampler.next_2d(), si.is_valid()
            )
            beta2 = beta * bsdf_weight
            ray2  = si.spawn_ray(si.to_world(bsdf_sample.wo))
            si2   = scene.ray_intersect(ray2, ray_flags=mi.RayFlags.All, coherent=False)
            # Force evaluation here: breaks the lazy graph so the model receives
            # concrete si2 data rather than a fused bsdf_sample+intersect+model kernel
            # that would take minutes to JIT-compile.
            dr.eval(si2)

            # Query cache at x2
            if getattr(self.model, '_is_nrc', False):
                cache2 = dr.maximum(beta2 * self.model(si2, ray2), mi.Color3f(0.0))
            else:
                gaussians_list, vmfs_list = self.model(si2)
                mixture = vapl_mixture_drjit(gaussians_list, vmfs_list, self.sweep_encoding)
                mixture.convolve(si2, ray2.d)
                Le2    = mi.Spectrum(0) if self.indirect_only else si2.emitter(scene).eval(si2)
                cache2 = dr.maximum(beta2 * (mixture.illumination + Le2), mi.Color3f(0.0))

            L += dr.select(si2.is_valid(), cache2, mi.Spectrum(0))
            return L, si.is_valid(), [], mi.Spectrum(0)

