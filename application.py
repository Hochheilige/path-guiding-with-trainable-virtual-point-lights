import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
import matplotlib.pyplot as plt
import wandb

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
               }
            }
            self.scene : mi.Scene = mi.load_dict(scene_dict)
        else:
            self.scene : mi.Scene = mi.load_file(config.scene)

        self.grid = vapl_grid_base.create_vapl_grid(config, self.scene.bbox().min, self.scene.bbox().max)
        self.loss_function = Loss(torch.nn.MSELoss())
        if self.config.grid.layout == "nrc":
            self.integrator = RHSIntegrator(self.grid, self.loss_function, True, True)
        else:
            self.integrator = RHSIntegrator(self.grid, self.loss_function, True)

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
            self.config.sweep_config = wandb.config
            self.epoch = self.config.sweep_config.epoch
            self.train()

    def train(self):
        if self.config.mode == "wandb":
            wandb.init(
                project="vapls-training",
                name=self.config.run_name,
                config=self.config
            )

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
        if (self.config.mode == "wandb" or self.config.mode == "sweep"):
            wandb.log({"loss": self.integrator.losses[-1].item(), "epoch": epoch})

        if (self.should_render(epoch)):
            with torch.no_grad():
                gaussians, vmfs = self.grid.get_gaussians_for_debug_render()

                mean = gaussians[:, :3]
                variance = gaussians[:, 3]
                amplitude = vmfs[:, 4:7]
                axis = vmfs[:, 1:4]

                h, w = image.shape[0], image.shape[1]

                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                ax[0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                ax[0].axis("off")

                self.debug_vapl_render(self.scene, mean, variance, amplitude,axis, h, w, ax[1])
                ax[1].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                ax[1].axis("off")

                if self.config.mode == "local":
                    ax[0].set_title(f"vapl render - epoch:{epoch}")
                    ax[1].set_title(f"vapl debug - epoch:{epoch}")
                    plt.show()
                else: # it may look not good on wandb
                    wandb.log({"vapl training": wandb.Image(fig)})

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
        variance = variance.cpu().detach().numpy().flatten()
        amplitude = amplitude.cpu().detach().numpy()
        means_ndc = world_to_ndc(scene, p)
        means_pix = ndc_to_pixel(means_ndc, h, w)

        amplitude_norm = amplitude / amplitude.max() if amplitude.max() != 0 else amplitude
        colors = amplitude_norm

        point_sizes = 10 * variance

        axis_nds = world_to_ndc(scene, axis.cpu().detach().numpy())
        axis_pix = ndc_to_pixel(axis_nds, h, w)

        dx = axis_pix.x - means_pix.x
        dy = axis_pix.y - means_pix.y

        ax.scatter(means_pix.x, means_pix.y, c=colors, cmap='coolwarm', marker='o', s=point_sizes)
        # TODO: figure out how to render arrows more correct
        #ax.quiver(means_pix.x, means_pix.y, dx, dy, angles='uv', color=colors, scale=1, scale_units='xy')
