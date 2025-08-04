import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
import matplotlib.pyplot as plt
import wandb
import gc

from integrator import *
from vapl_grid import *

class Application:
    def __init__(self, config):
        self.config = config

        if config.scene == "cornell box":
            # cornell box with specular sphere
            scene_dict = mi.cornell_box()
            scene_dict['sphere'] = {
               'type': 'sphere',
               'radius': 0.4,
               'center': [0, 0.2, 0],
               'bsdf': {
                   'type': 'roughconductor',
                   'distribution': 'ggx',
                   'alpha_u': 0.5,
                   'alpha_v': 0.1
               },
            }
            self.scene : mi.Scene = mi.load_dict(scene_dict)
        else:
            self.scene : mi.Scene = mi.load_file(config.scene)

        self.grid = vapl_grid_base.create_vapl_grid(config, self.scene.bbox().min, self.scene.bbox().max)
        self.loss_function = Loss(relativeL2_luminance_tiny_cuda_nn)
        self.integrator = RHSIntegrator(self.grid, self.loss_function, True)
        self.integrator.set_depth(self.config.depth)

        if config.mode == "wandb":
            wandb.login()

        if config.mode != "sweep":
            self.epoch = self.config.epoch
            image = mi.render(self.scene, spp=128)
            fig, ax = plt.subplots()
            ax.imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
            ax.axis("off")
            ax.set_title(f"Path-traced image")
            plt.show()

    def sweep(self):
        if self.config.mode == "sweep":
            wandb.init(project="vapls-parameters-encodings-search")
            self.recreate()
            self.train()

    def recreate(self):
        del self.grid
        del self.integrator
        del self.scene
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        dr.kernel_history_clear()
        dr.flush_malloc_cache()
        dr.malloc_clear_statistics()
        dr.flush_kernel_cache()
        if self.config.scene == "cornell box":
            # cornell box with specular sphere
            scene_dict = mi.cornell_box()
            scene_dict['sphere'] = {
               'type': 'sphere',
               'radius': 0.4,
               'center': [0, 0.2, 0],
               'bsdf': {
                   'type': 'roughconductor',
                   'distribution': 'ggx',
                   'alpha_u': 0.5,
                   'alpha_v': 0.1
               },
            }
            self.scene : mi.Scene = mi.load_dict(scene_dict)
        else:
            self.scene : mi.Scene = mi.load_file(self.config.scene)
        self.config.sweep_config = wandb.config
        self.epoch = self.config.sweep_config.epoch
        self.grid = vapl_grid_base.create_vapl_grid(self.config, self.scene.bbox().min, self.scene.bbox().max)
        self.integrator = RHSIntegrator(self.grid, self.loss_function, True)
        self.integrator.set_depth(self.config.depth)
        self.integrator.set_config(self.config.sweep_config.vmf_axis_encoding)

    def train(self):
        if (self.config.mode != "sweep"):
            self.integrator.set_config(self.config.grid.vmf_axis_encoding)

        if self.config.mode == "wandb":
            if self.config.wandb_group == "default":
                wandb.init(
                    project="vapls-training",
                    name=self.config.run_name,
                    config=self.config
                )
            else:
                wandb.init(
                    project="vapls-training",
                    name=self.config.run_name,
                    group=self.config.wandb_group,
                    config=self.config
                )

        # draw reference image
        self.integrator.set_train(False)
        image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator)
        fig, ax = plt.subplots()
        ax.imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax.axis("off")
        ax.set_title(f"Reference image")
        plt.show()
        self.integrator.set_train(True)

        for epoch in range(self.epoch):
            self.integrator.epoch = epoch
            image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator)
            self.render(epoch, image)

        if self.config.mode == "wandb":
            wandb.finish()

    def render_trained(self, spp):
        self.integrator.set_train(False)
        image = mi.render(self.scene, spp=spp, integrator=self.integrator)
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax[0].axis("off")
        ax[0].set_title(f"Path-traced image with VAPL")

        image = mi.render(self.scene, spp=spp)
        ax[1].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax[1].axis("off")
        ax[1].set_title(f"Path-traced image")
        plt.show()

    def render(self, epoch, image):
        if self.config.mode in ["wandb", "sweep"]:
            wandb.log({"loss": self.integrator.losses[-1].item(), "epoch": epoch})

        if not self.should_render(epoch):
            return

        with torch.no_grad():
            gaussians_list, vmfs_list = self.grid.get_gaussians_for_debug_render()
            h, w = image.shape[0], image.shape[1]

            n_levels = self.config.grid.n_levels
            total_plots = n_levels + 1

            gaussians_list = gaussians_list[:n_levels]
            vmfs_list = vmfs_list[:n_levels]

            fig, axs = plt.subplots(1, total_plots, figsize=(6 * total_plots, 6))

            axs[0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
            axs[0].axis("off")
            axs[0].set_title(f"vapl render - epoch:{epoch}")

            for level, (gaussians, vmfs) in enumerate(zip(gaussians_list, vmfs_list)):
                mean = gaussians[:, :3]
                variance = gaussians[:, 3]
                amplitude = vmfs[:, 4:7]
                axis = vmfs[:, 1:4]

                self.debug_vapl_render(self.scene, mean, variance, amplitude, axis, h, w, axs[level + 1])
                axs[level + 1].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                axs[level + 1].axis("off")
                axs[level + 1].set_title(f"level {level}")

            plt.tight_layout()

            if self.config.mode == "local":
                plt.show()
            else:
                wandb.log({f"vapl training": wandb.Image(fig)})

    def should_render(self, epoch):
        if self.config.mode == "sweep":
            return False

        if epoch < 50:
            return epoch % 5 == 0
        elif epoch < 500:
            return epoch % 20 == 0
        elif epoch < 2000:
            return epoch % 100 == 0
        else:
            return epoch % 250 == 0

    def debug_vapl_render(self, scene, pos, variance, amplitude, axis, h, w, ax):
        p = pos.cpu().detach().numpy()
        amplitude = amplitude.cpu().detach().numpy()
        variance = variance.cpu().detach().numpy().flatten()

        means_ndc = world_to_ndc(scene, p)
        means_pix = ndc_to_pixel(means_ndc, h, w)

        colors = amplitude / (amplitude.max() + 1e-8)

        sigma = np.sqrt(variance)
        point_sizes = (sigma / sigma.max() * 50)

        axis = axis.cpu().detach().numpy()
        axis_ndc = world_to_ndc(scene, p + axis * 0.05)
        axis_pix = ndc_to_pixel(axis_ndc, h, w)

        dx = axis_pix.x - means_pix.x
        dy = axis_pix.y - means_pix.y

        ax.scatter(means_pix.x, means_pix.y, c=colors, s=point_sizes, marker='o')
        # TODO: figure out how to render arrows more correct
        #ax.quiver(means_pix.x, means_pix.y, dx, dy, angles='uv', scale=1, scale_units='xy', color=colors)

