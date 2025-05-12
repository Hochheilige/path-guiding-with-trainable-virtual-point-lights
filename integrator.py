import drjit as dr
import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
from mitsuba.python.ad.integrators.common import ADIntegrator, mis_weight

import inspect
import torch
torch.autograd.set_detect_anomaly(True)

from vapl_utils import vapl_mixture

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

def weighted_loss(real, predicted, weight):
    eps = 0.01
    mse = (real - predicted) ** 2
    norm_factor = (weight * (predicted ** 2).detach() + eps)
    return (mse / norm_factor).mean()

def relativeL2(ref, prediction, pdf):
    eps = 1e-2
    div = prediction.detach()
    denominator = torch.mean(div, dim=1).view(-1,1)**2 +eps

    rL2 = (pdf*((prediction - ref))**2) / denominator
    rL2 = rL2[~(torch.isinf(rL2) | torch.isnan(rL2))]
    rL2 = rL2.mean()
    return rL2

def compute_luminance(rgb):
    coeffs = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device).view(1, 3)
    return (rgb * coeffs).sum(dim=1, keepdim=True)

def relative_l2_loss_with_luminance(target, prediction):
    epsilon = 0.01
    prediction_detached = prediction.detach()
    error_squared = (target - prediction) ** 2
    luminance_sq = compute_luminance(prediction_detached) ** 2 + epsilon
    relative_error = error_squared / luminance_sq
    return relative_error.mean()

class RHSIntegrator(ADIntegrator):
    def __init__(self, model, loss_function : Loss, train, is_nrc = False, sweep_encoding = None, props=mi.Properties()):
        super().__init__(props)
        self.train = train
        self.model = model
        self.losses = []
        self.loss_function = loss_function
        self.sweep_encoding = sweep_encoding
        self.is_nrc = is_nrc

    def set_train(self, train):
        self.train = train

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

    def sample_training(self, scene: mi.Scene, sampler: mi.Sampler, ray: mi.Ray3f, depth: mi.UInt32):
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

        if self.is_nrc != True:
            gaussians, vmfs = self.model(si)
            mixture = vapl_mixture(gaussians, vmfs)
            mixture.convolve_with_bsdf(si, ray.d)
        else:
            def encode_position(position):
                return encode_frequency(position)

            def encode_roughness(roughness, resolution=4):
                r = 1.0 - torch.exp(-roughness)
                centers = torch.linspace(0.0, 1.0, resolution, device=r.device)
                r_expanded = r.unsqueeze(-1)
                encoding = torch.exp(-((r_expanded - centers) ** 2) * 40.0)
                return encoding.view(-1, resolution)

            def encode_reflectance(reflectance):
                return reflectance

            def encode_frequency(x):
                freqs = []
                for i in range(6):
                    freq = 2 ** i
                    freqs.append(torch.sin(freq * x))
                    freqs.append(torch.cos(freq * x))
                return torch.cat(freqs, dim=-1)

            def encode_spherical_like_one_blob(x, resolution=8):
                x = torch.nn.functional.normalize(x, p=2, dim=1)
                direction_scalar = x[:, 0] * 0.2989 + x[:, 1] * 0.5870 + x[:, 2] * 0.1140
                direction_scalar = direction_scalar.unsqueeze(-1)

                centers = torch.linspace(-1.0, 1.0, resolution, device=x.device)
                gauss = torch.exp(-((direction_scalar - centers) ** 2) * 40.0)
                return gauss

            position = si.p.torch().permute(1, 0)
            normal = si.sh_frame.n.torch().permute(1, 0)
            direction = -si.wi.torch().permute(1, 0)

            view_dir_normalize = (torch.nn.functional.normalize(ray.d.torch().permute(1, 0), p=2, dim=1, eps=1e-6))
            wo_world = -view_dir_normalize

            bsdf = si.bsdf()
            wo_ts = si.sh_frame.to_local(mi.Vector3f(wo_world.permute(1, 0)))
            ctx_diffuse = mi.BSDFContext()
            ctx_diffuse.type_mask = mi.BSDFFlags.Diffuse
            ctx_specular = mi.BSDFContext()
            ctx_specular.type_mask = mi.BSDFFlags.Glossy
            roughness = bsdf.eval_attribute_1("alpha", si).torch().unsqueeze(1)

            diffuse: mi.Spectrum = bsdf.eval(ctx_diffuse, si, wo_ts)
            specular: mi.Spectrum = bsdf.eval(ctx_specular, si, wo_ts)
            diffuse_tensor: torch.Tensor = diffuse.torch().permute(1, 0)
            specular_tensor: torch.Tensor = specular.torch().permute(1, 0)

            encoded_position = encode_position(position)
            #print(encoded_position.shape)
            encoded_normal = encode_spherical_like_one_blob(normal)
            #print(encoded_normal.shape)
            encoded_direction = encode_spherical_like_one_blob(direction)
            #print(encoded_direction.shape)
            encoded_roughness = encode_roughness(roughness)
            #print(encoded_roughness.shape)
            encoded_diffuse = encode_reflectance(diffuse_tensor)
            #print(encoded_diffuse.shape)
            encoded_specular = encode_reflectance(specular_tensor)
            #print(encoded_specular.shape)


            inputs = torch.cat([
                encoded_position,
                encoded_normal,
                encoded_direction,
                encoded_roughness,
                encoded_diffuse,
                encoded_specular
            ], dim=1).cuda()

            vapl_l = torch.relu((diffuse_tensor + specular_tensor).to(dtype=torch.float32) * self.model(inputs).to(dtype=torch.float32))

        # update si and bsdf with the first non-specular ones
        si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)

        # TODO: make this better but not critical
        bss = []

        for depth in range(4):
            if (depth > 0):
                si = scene.ray_intersect(
                    ray, ray_flags=mi.RayFlags.All, coherent=(depth==0)
                )

                # update si and bsdf with the first non-specular ones
                si, β, _ = first_non_specular_or_null_si(scene, si, sampler, β)


            L, bs, β = render_rhs_original(scene, si, sampler, β)
            ray = si.spawn_ray(si.to_world(bs.wo))

            res_l = res_l + (L)
            bss.append(bs)

        if self.is_nrc != True:
            vapl_l = mixture.illumination

        return res_l, vapl_l, bss[0].pdf.torch().unsqueeze(-1), si


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
            L, L_vapl, weight, si = self.sample_training(scene, sampler, ray, depth)

            L_tensor = torch.from_numpy(L.numpy()).to("cuda").T

            loss : torch.Tensor = self.loss_function(L_vapl, L_tensor, weight)
            self.losses.append(loss.detach().cpu())
            self.model.optimizer.zero_grad()
            loss.backward()
            self.model.optimizer.step()

            torch.cuda.empty_cache()
            return L_vapl.permute(1, 0), si.is_valid(), [], mi.Spectrum(0)
        else:
            L, si = self.sample_using_vapls(mode, scene, sampler, ray, depth, δL, δaovs, state_in, active)
            return L, si.is_valid(), [], mi.Spectrum(0)

