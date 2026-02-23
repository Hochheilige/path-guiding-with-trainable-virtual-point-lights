import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

import inspect
import torch
torch.autograd.set_detect_anomaly(True)

from vapl_utils import vapl_mixture
from vapl_utils_drjit import vapl_mixture_drjit

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

# =============================================================================
# Loss functions — Torch
# =============================================================================

class Loss():
    def __init__(self, loss_fn):
        self.loss_fn = loss_fn
        self.num_params = len(inspect.signature(loss_fn).parameters)

    def __call__(self, pred, target, weight = None):
        if self.num_params == 2:
            result = self.loss_fn(pred, target)
        elif self.num_params == 3:
            result = self.loss_fn(pred, target, weight)
        return result

def weighted_loss(predicted, real, weight):
    eps = 0.01
    mse = (real - predicted) ** 2
    norm_factor = (weight * (predicted ** 2).detach() + eps)
    return (mse / norm_factor).mean()

def relativeL2(prediction, ref, pdf):
    eps = 1e-2
    div = prediction.detach()
    denominator = torch.mean(div, dim=1).view(-1,1)**2 +eps

    rL2 = (pdf*((prediction - ref))**2) / denominator
    rL2 = rL2[~(torch.isinf(rL2) | torch.isnan(rL2))]
    rL2 = rL2.mean()
    return rL2

def relativeL2_luminance(prediction, reference):
    eps=1e-2
    # luminance: 0.2126 R + 0.7152 G + 0.0722 B
    luminance = (prediction.detach() * torch.tensor([0.2126, 0.7152, 0.0722], device=prediction.device)).sum(dim=1, keepdim=True)
    denom = luminance ** 2 + eps

    loss = ((prediction - reference) ** 2) / denom
    loss = loss[~torch.isnan(loss) & ~torch.isinf(loss)]
    return loss.mean()

def relativeL2_luminance_tiny_cuda_nn(pred, target, pdf=None):
    loss_scale=1.0
    eps=1e-2

    assert pred.shape == target.shape, "Prediction and target must have the same shape"
    N, C = pred.shape
    device = pred.device

    rgb = pred[:, 0:3]
    luminance = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).detach()
    denom = luminance ** 2 + eps

    if pdf is None:
        pdf = torch.ones(N, device=device)
    else:
        pdf = pdf.detach().clamp(min=1e-6)

    diff = pred - target
    sq_error = diff ** 2

    denom = denom.view(-1, 1)
    pdf = pdf.view(-1, 1)
    loss = sq_error / denom / pdf

    valid = torch.isfinite(loss)
    if not torch.all(valid):
        loss = loss[valid]

    if loss.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss_scale * loss.mean()

def torch_mse(pred, target):
    diff = pred - target
    loss = (diff ** 2)
    valid = torch.isfinite(loss)
    if not torch.all(valid):
        loss = loss[valid]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss.mean()

def torch_log_relative(pred, target):
    """Log-space loss — compresses dynamic range so dim indirect light
    contributes as much as bright direct light."""
    eps = 1e-4
    log_pred = torch.log1p(pred.clamp(min=0) / eps)
    log_target = torch.log1p(target.clamp(min=0) / eps)
    loss = (log_pred - log_target) ** 2
    valid = torch.isfinite(loss)
    if not torch.all(valid):
        loss = loss[valid]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss.mean()

def torch_smape(pred, target):
    """Symmetric Mean Absolute Percentage Error — balanced for both
    bright and dim regions, no luminance denominator bias."""
    eps = 1e-2
    diff = torch.abs(pred - target)
    denom = torch.abs(pred.detach()) + torch.abs(target) + eps
    loss = diff / denom
    valid = torch.isfinite(loss)
    if not torch.all(valid):
        loss = loss[valid]
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return loss.mean()


# =============================================================================
# Loss functions — DrJIT
# =============================================================================

class DrJITLoss():
    """Unified loss class for the DrJIT training path.

    Usage:
        loss_fn = DrJITLoss("relative_l2")   # or "mse", "log_relative", "smape"
        loss = loss_fn(pred, target)
    """
    LOSSES = {
        "relative_l2": "_relative_l2",
        "mse": "_mse",
        "log_relative": "_log_relative",
        "smape": "_smape",
    }

    def __init__(self, name="relative_l2"):
        if name not in self.LOSSES:
            raise ValueError(f"Unknown loss '{name}'. Choose from: {list(self.LOSSES.keys())}")
        self.name = name
        self._fn = getattr(self, self.LOSSES[name])

    def __call__(self, pred, target):
        return self._fn(pred, target)

    @staticmethod
    def _filter(loss_per_ray):
        valid = dr.isfinite(loss_per_ray)
        loss_per_ray = dr.select(valid, loss_per_ray, mi.Float(0.0))
        n = mi.Float(dr.width(loss_per_ray))
        return dr.sum(loss_per_ray) / dr.maximum(n, mi.Float(1.0))

    @staticmethod
    def _relative_l2(pred, target):
        """Relative L2 luminance loss (original)."""
        eps = 1e-2
        pred_d = dr.detach(pred)
        luminance = 0.2126 * pred_d.x + 0.7152 * pred_d.y + 0.0722 * pred_d.z
        denom = luminance * luminance + eps
        sq_error = dr.squared_norm(pred - target)
        return DrJITLoss._filter(sq_error / denom)

    @staticmethod
    def _mse(pred, target):
        """Plain mean squared error."""
        sq_error = dr.squared_norm(pred - target)
        return DrJITLoss._filter(sq_error)

    @staticmethod
    def _log_relative(pred, target):
        """Log-space loss — compresses dynamic range so dim indirect light
        contributes as much as bright direct light."""
        eps = 1e-4
        log_pred = dr.log(1.0 + dr.maximum(pred, mi.Spectrum(0)) / eps)
        log_target = dr.log(1.0 + dr.maximum(target, mi.Spectrum(0)) / eps)
        diff = log_pred - log_target
        sq_error = dr.squared_norm(diff)
        return DrJITLoss._filter(sq_error)

    @staticmethod
    def _smape(pred, target):
        """Symmetric Mean Absolute Percentage Error — balanced for both
        bright and dim regions."""
        eps = 1e-2
        pred_d = dr.detach(pred)
        diff = dr.abs(pred - target)
        denom_r = dr.abs(pred_d.x) + dr.abs(target.x) + eps
        denom_g = dr.abs(pred_d.y) + dr.abs(target.y) + eps
        denom_b = dr.abs(pred_d.z) + dr.abs(target.z) + eps
        loss_per_ray = diff.x / denom_r + diff.y / denom_g + diff.z / denom_b
        return DrJITLoss._filter(loss_per_ray)


# Legacy alias so existing code keeps working
def compute_drjit_loss(pred, target):
    return DrJITLoss._relative_l2(pred, target)


class RHSIntegrator(ADIntegrator):
    def __init__(self, model, loss_function : Loss, train, sweep_encoding = None,
                 drjit_loss_name="relative_l2", indirect_only=False, props=mi.Properties()):
        super().__init__(props)
        self.train = train
        self.model = model
        self.losses = []
        self.loss_function = loss_function              # torch path
        self.drjit_loss_function = DrJITLoss(drjit_loss_name)  # drjit path
        self.sweep_encoding = sweep_encoding
        self.indirect_only = indirect_only
        self.depth = 1

    def set_train(self, train):
        self.train = train

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

                    # GT: path trace from the first diffuse hit (no gradients)
                    with dr.suspend_grad():
                        gt_light = path_trace_from_si(scene, si, sampler, beta,
                                                      max_depth=self.depth, rr_depth=2,
                                                      indirect_only=self.indirect_only)

                    # VAPL prediction at the same si (gradients flow through model)
                    gaussians_list, vmfs_list = self.model(si)
                    for mean, var in gaussians_list:
                        dr.eval(mean, var)
                    for sh, ax, amp in vmfs_list:
                        dr.eval(sh, ax, amp)
                    mixture = vapl_mixture_drjit(gaussians_list, vmfs_list, self.sweep_encoding)
                    mixture.convolve(si, ray_d.d)
                    # indirect_only: exclude Le so prediction matches GT (no direct emission)
                    Le         = mi.Spectrum(0) if self.indirect_only else si.emitter(scene).eval(si)
                    vapl_light = beta * (mixture.illumination + Le)

                    loss = self.drjit_loss_function(vapl_light, gt_light)

                    # Mixed-precision: scale loss before backward, then scaled step
                    dr.backward(self.model.scaler.scale(loss))
                    self.model.optimizer.step()

                self.losses.append(float(loss[0]))

                return vapl_light, si.is_valid(), [], mi.Spectrum(0)
            else:
                # PyTorch training path (original)
                vapl_light, gt_light, si = self.sample_training(scene, sampler, ray, depth)
                GT_Light = torch.from_numpy(gt_light.numpy()).to("cuda").T

                loss : torch.Tensor = self.loss_function(vapl_light, GT_Light, None)
                self.losses.append(loss.detach().cpu())
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

                torch.cuda.empty_cache()
                return vapl_light.permute(1, 0), si.is_valid(), [], mi.Spectrum(0)
        else:
            # Inference: same path as training GT for consistency
            ray_d  = mi.Ray3f(dr.detach(ray))
            beta   = mi.Spectrum(1)
            si_raw = scene.ray_intersect(ray_d, ray_flags=mi.RayFlags.All, coherent=(depth == 0))
            si, beta, _ = first_non_specular_or_null_si(scene, si_raw, sampler, beta)
            L = path_trace_from_si(scene, si, sampler, beta,
                                   max_depth=self.depth, rr_depth=2,
                                   indirect_only=self.indirect_only)
            return L, si.is_valid(), [], mi.Spectrum(0)

