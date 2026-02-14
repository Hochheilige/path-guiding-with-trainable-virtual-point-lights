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


def render_rhs(scene : mi.Scene, si : mi.SurfaceInteraction3f, sampler, β):
    with dr.suspend_grad():
        # All the stuff from original render_rhs function
        bsdf_ctx = mi.BSDFContext()
        depth = mi.UInt32(0)
        L = mi.Spectrum(0)
        η = mi.Float(1)
        prev_si = dr.zeros(mi.SurfaceInteraction3f)
        prev_bsdf_pdf = mi.Float(1.0)
        prev_bsdf_delta = mi.Bool(True)

        bsdf = si.bsdf()
        Le = β * si.emitter(scene).eval(si)

        # emitter sampling
        active_next = si.is_valid()
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)

        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )
        active_em &= (ds.pdf != 0.0)

        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        Lr_dir = β * mis_em * bsdf_value_em * em_weight

        # bsdf sampling
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
        )

        # update
        L = L + Le + Lr_dir

        #η = bsdf_sample.eta
        β *= bsdf_weight

        # prev_si = dr.detach(si, True)
        # prev_bsdf_pdf = bsdf_sample.pdf
        # prev_bsdf_delta = mi.has_flag(bsdf_sample.sampled_type, mi.BSDFFlags.Delta)

        # si = scene.ray_intersect(ray, ray_flags=mi.RayFlags.All, coherent=True)
        # ds = mi.DirectionSample3f(scene, si=si, ref=prev_si)

        # mis = mis_weight(
        #     prev_bsdf_pdf,
        #     scene.pdf_emitter_direction(prev_si, ds, ~prev_bsdf_delta),
        # )

        # si, β2, null_face = first_non_specular_or_null_si(scene, si, sampler)
        # β *= β2

        # L += β * mis * si.emitter(scene).eval(si)

        return L, β


def render_rhs_original(scene, si, sampler, β):
    with dr.suspend_grad():
        bsdf_ctx = mi.BSDFContext()
        L = mi.Spectrum(0)
        bsdf = si.bsdf()
        Le = β * si.emitter(scene).eval(si)

        # emitter sampling
        active_next = si.is_valid()
        active_em = active_next & mi.has_flag(bsdf.flags(), mi.BSDFFlags.Smooth)
        ds, em_weight = scene.sample_emitter_direction(
            si, sampler.next_2d(), True, active_em
        )

        active_em &= (ds.pdf != 0.0)
        wo = si.to_local(ds.d)
        bsdf_value_em, bsdf_pdf_em = bsdf.eval_pdf(bsdf_ctx, si, wo, active_em)
        mis_em = dr.select(ds.delta, 1, mis_weight(ds.pdf, bsdf_pdf_em))
        Lr_dir = β * mis_em * bsdf_value_em * em_weight

        # bsdf sampling
        bsdf_sample, bsdf_weight = bsdf.sample(
            bsdf_ctx, si, sampler.next_1d(), sampler.next_2d(), active_next
        )
        # update
        L = L + Le + Lr_dir

    return L, bsdf_sample, β*bsdf_weight

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


def compute_drjit_loss(pred, target):
    """Relative L2 luminance loss computed entirely in DrJIT.

    Args:
        pred: mi.Color3f — predicted illumination
        target: mi.Spectrum — ground truth from path tracing
    Returns:
        mi.Float — scalar loss value
    """
    eps = 1e-2
    # luminance of prediction (detached for denominator)
    pred_d = dr.detach(pred)
    luminance = 0.2126 * pred_d.x + 0.7152 * pred_d.y + 0.0722 * pred_d.z
    denom = luminance * luminance + eps

    diff = pred - target
    sq_error = dr.squared_norm(diff)

    loss_per_ray = sq_error / denom

    # Filter out non-finite values
    valid = dr.isfinite(loss_per_ray)
    loss_per_ray = dr.select(valid, loss_per_ray, mi.Float(0.0))

    n = mi.Float(dr.width(loss_per_ray))
    return dr.sum(loss_per_ray) / dr.maximum(n, mi.Float(1.0))


class RHSIntegrator(ADIntegrator):
    def __init__(self, model, loss_function : Loss, train, sweep_encoding = None, props=mi.Properties()):
        super().__init__(props)
        self.train = train
        self.model = model
        self.losses = []
        self.loss_function = loss_function
        self.sweep_encoding = sweep_encoding
        self.depth = 1
        self.gt_light = mi.Spectrum(1)
        self.vapl_ratio = 0.0  # 0 = pure PT, 1 = pure VAPL

    def set_train(self, train):
        self.train = train

    def set_vapl_ratio(self, ratio):
        self.vapl_ratio = ratio

    def set_depth(self, depth):
        self.depth = depth

    # Basics for Path-tracing using trained vapls
    @dr.syntax
    def sample_using_vapls(self,
               mode: dr.ADMode,
               scene: mi.Scene,
               sampler: mi.Sampler,
               ray: mi.Ray3f,
               depth: mi.UInt32,
               δL,
               δaovs,
               state_in,
               active):
        w, h = list(scene.sensors()[0].film().size())
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)

        ray = mi.Ray3f(dr.detach(ray))
        max_depth = 4
        self.losses = []
        si = None

        for depth in range(max_depth):
            #print("iteration: ", depth)
            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=(depth == 0)
            )

            # update si and bsdf with the first non-specular ones
            # LOOKS LIKE this funciton could make things worse because
            # using it instead of directly use our new direction we would sample BSDF
            # if original si gives smooth or null surface
            #si, β, _ = first_non_specular_or_null_si(scene, si, sampler)

            # get the vapl mixture for this intersection
            gaussians, vmfs = self.model(si)
            mixture = vapl_mixture(gaussians, vmfs)
            mixture.sample_vapl(si, ray.d)

            # Calculating new sample direction

            # 1st option - Sample direction from sampled vapl light lobe
            sampled_dir : torch.Tensor = mixture.sample_from_current_ligth_lobe_vmf()
            print(sampled_dir)

            # 2nd option - Sample direction according to BSDF x vapl convolution
            # Specular BSDF - Anisotropic Spherical Gaussian
            # Diffuse BSDF  - Cosine Lobe

            # FIXME:
            # Looks like this approach works worse,
            # but probably because not totally correct previous calculations
            #sampled_dir :torch.Tensor = mixture.sample_from_current_bsdf_light_lobe_vmf()

            Li, β = render_rhs(scene, si, sampler, β)

            # Use new direction from vapl mixture to generate next ray
            new_dir = mi.cuda_ad_rgb.Vector3f(sampled_dir)
            ray = si.spawn_ray(new_dir)

            # L_tensor = torch.from_numpy(Li.numpy()).to("cuda").T
            # light_from_vapl = mixture.illumination

            # mse_loss_func = torch.nn.MSELoss()
            # loss = mse_loss_func(light_from_vapl, L_tensor)
            # self.losses.append(loss.item())
            # loss.backward()
            # self.model.sg_optimizer.step()
            # self.model.vmf_optimizer.step()
            # self.model.sg_optimizer.zero_grad()
            # self.model.vmf_optimizer.zero_grad()

            L += Li
        return L, si

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

        # Compute ground truth at the same surface interaction
        with dr.suspend_grad():
            gt_light = mi.Spectrum(0)
            β_gt = mi.Spectrum(β)
            si_gt = si
            for d in range(self.depth):
                if d > 0:
                    si_gt = scene.ray_intersect(
                        ray_gt, ray_flags=mi.RayFlags.All, coherent=False
                    )
                    si_gt, β_gt, _ = first_non_specular_or_null_si(scene, si_gt, sampler, β_gt)
                L, bs, β_gt = render_rhs_original(scene, si_gt, sampler, β_gt)
                ray_gt = si_gt.spawn_ray(si_gt.to_world(bs.wo))
                gt_light += L

        # Compute VAPL prediction at the same si
        gaussians, vmfs = self.model(si)
        mixture = vapl_mixture(gaussians, vmfs, self.sweep_encoding)
        mixture.convolve(si, ray.d)

        Le = si.emitter(scene).eval(si)
        Le_torch = Le.torch().permute(1, 0)
        vapl_l = mixture.illumination + Le_torch

        return vapl_l, gt_light, si

    def sample_training_drjit(self, scene, sampler, ray, depth):
        """DrJIT-native training path — no torch conversions."""
        ray = mi.Ray3f(dr.detach(ray))
        beta = mi.Spectrum(1)

        si = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=(depth == 0)
        )

        si, beta, _ = first_non_specular_or_null_si(scene, si, sampler, beta)
        si.compute_uv_partials(ray)

        # Compute ground truth at the same surface interaction
        with dr.suspend_grad():
            gt_light = mi.Spectrum(0)
            beta_gt = mi.Spectrum(beta)
            si_gt = si
            for d in range(self.depth):
                if d > 0:
                    si_gt = scene.ray_intersect(
                        ray_gt, ray_flags=mi.RayFlags.All, coherent=False
                    )
                    si_gt, beta_gt, _ = first_non_specular_or_null_si(scene, si_gt, sampler, beta_gt)
                L, bs, beta_gt = render_rhs_original(scene, si_gt, sampler, beta_gt)
                ray_gt = si_gt.spawn_ray(si_gt.to_world(bs.wo))
                gt_light += L

        gaussians_list, vmfs_list = self.model(si)

        # Evaluate grid outputs to free intermediate JIT memory
        # before the mixture convolution builds more AD nodes
        for mean, var in gaussians_list:
            dr.eval(mean, var)
        for sh, ax, amp in vmfs_list:
            dr.eval(sh, ax, amp)

        mixture = vapl_mixture_drjit(gaussians_list, vmfs_list, self.sweep_encoding)
        mixture.convolve(si, ray.d)

        # Add direct emission so emitter surfaces aren't forced to zero.
        # Le is constant w.r.t. grid params — no effect on backward pass,
        # but removes impossible gradients at emitter surfaces.
        Le = si.emitter(scene).eval(si)

        return mixture.illumination + Le, gt_light, si

    def sample_training_ref(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, depth: mi.UInt32):
        w, h = list(scene.sensors()[0].film().size())
        L = mi.Spectrum(0)
        β = mi.Spectrum(1)
        bsdf_ctx = mi.BSDFContext()

        ray = mi.Ray3f(dr.detach(ray))
        vapl_l = torch.zeros((w*h, 3), device="cuda")
        res_l = mi.Spectrum(0)

        si = scene.ray_intersect(
            ray, ray_flags=mi.RayFlags.All, coherent=(depth==0)
        )
        si.compute_uv_partials(ray)

        # update si and bsdf with the first non-specular ones
        si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)

        for depth in range(self.depth):
            if (depth > 0):
                si = scene.ray_intersect(
                    ray, ray_flags=mi.RayFlags.All, coherent=(depth==0)
                )

                # update si and bsdf with the first non-specular ones
                si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)


            L, bs, β = render_rhs_original(scene, si, sampler, β)
            ray = si.spawn_ray(si.to_world(bs.wo))

            res_l = res_l + (L)

        self.gt_light = res_l
        return res_l, si

    def sample_hybrid(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, depth: mi.UInt32):
        """Hybrid inference: blend VAPL cache with path tracing.

        Per-ray random split: vapl_ratio fraction of rays use VAPLs,
        the rest use standard path tracing. Each path is weighted by
        1/probability to keep the estimator consistent.
        """
        with dr.suspend_grad():
            ray = mi.Ray3f(dr.detach(ray))
            β = mi.Spectrum(1)

            si = scene.ray_intersect(
                ray, ray_flags=mi.RayFlags.All, coherent=(depth == 0)
            )
            si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)
            si.compute_uv_partials(ray)

            # Per-ray decision
            use_vapl = sampler.next_1d() < self.vapl_ratio

            # --- Path trace path ---
            res = mi.Spectrum(0)
            β_pt = mi.Spectrum(β)
            si_pt = si
            for d in range(self.depth):
                if d > 0:
                    si_pt = scene.ray_intersect(
                        ray_pt, ray_flags=mi.RayFlags.All, coherent=False
                    )
                    si_pt, β_pt, _ = first_non_specular_or_null_si(scene, si_pt, sampler, β_pt)

                if d > 1:
                    # --- VAPL cache path ---
                    gaussians_list, vmfs_list = self.model(si)

                    # Evaluate grid outputs to free intermediate JIT memory
                    # before the mixture convolution builds more AD nodes
                    for mean, var in gaussians_list:
                        dr.eval(mean, var)
                    for sh, ax, amp in vmfs_list:
                        dr.eval(sh, ax, amp)

                    mixture = vapl_mixture_drjit(gaussians_list, vmfs_list, self.sweep_encoding)
                    mixture.convolve(si, ray.d)
                    illum = mixture.illumination
                    res += β_pt * illum
                else:
                    L, bs, β_pt = render_rhs_original(scene, si_pt, sampler, β_pt)
                    res += β_pt * L

                ray_pt = si_pt.spawn_ray(si_pt.to_world(bs.wo))
                

            # # Blend with inverse-probability weighting
            # L = dr.select(
            #     use_vapl,
            #     L_vapl / self.vapl_ratio,
            #     L_pt / (1.0 - self.vapl_ratio),
            # )

        return res, si

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
                # DrJIT-native training path
                # ADIntegrator.render() wraps sample() in dr.suspend_grad(),
                # which suppresses DrJIT AD. We must resume it explicitly.
                with dr.resume_grad():
                    vapl_light, gt_light, si = self.sample_training_drjit(scene, sampler, ray, depth)

                    loss = compute_drjit_loss(vapl_light, gt_light)

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
            if self.vapl_ratio > 0:
                L, si = self.sample_hybrid(scene, sampler, ray, depth)
            else:
                L, si = self.sample_training_ref(scene, sampler, ray, depth)
            return L, si.is_valid(), [], mi.Spectrum(0)

