import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from .base_app import BaseApp


class WandbApp(BaseApp):
    def __init__(self, config):
        super().__init__(config)
        self.epoch = config.epoch
        wandb.login()

    def train(self, _skip_wandb_init=False):
        if not _skip_wandb_init:
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

        self.integrator.set_config(self.config.grid.vmf_axis_encoding)

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
            self._render_epoch(epoch, image)

        if not _skip_wandb_init:
            wandb.finish()

    def _render_epoch(self, epoch, image):
        last_loss = self.integrator.losses[-1]
        loss_val = last_loss.item() if hasattr(last_loss, 'item') else float(last_loss)
        wandb.log({"loss": loss_val, "epoch": epoch})

        if not self.should_render(epoch):
            return

        with torch.no_grad():
            gaussians_list, vmfs_list = self.grid.get_gaussians_for_debug_render()
            h, w = image.shape[0], image.shape[1]

            gaussians_list = gaussians_list[:self.config.grid.n_levels]
            vmfs_list = vmfs_list[:self.config.grid.n_levels]
            n_levels = len(gaussians_list)
            total_plots = n_levels + 1

            fig, axs = plt.subplots(1, total_plots, figsize=(6 * total_plots, 6), squeeze=False)
            axs = axs[0]

            axs[0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
            axs[0].axis("off")
            axs[0].set_title(f"vapl render - epoch:{epoch}")

            for level, (gaussians, vmfs) in enumerate(zip(gaussians_list, vmfs_list)):
                mean = gaussians[:, :3]
                variance = gaussians[:, 3]
                amplitude = vmfs[:, 4:7]
                axis = vmfs[:, 1:4]

                axs[level + 1].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                self.debug_vapl_render(self.scene, mean, variance, amplitude, axis, h, w, axs[level + 1])
                axs[level + 1].axis("off")
                axs[level + 1].set_title(f"level {level}")

            plt.tight_layout()
            wandb.log({f"vapl training": wandb.Image(fig)})
