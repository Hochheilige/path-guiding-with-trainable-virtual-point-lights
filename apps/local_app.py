import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt
import torch

from .base_app import BaseApp


class LocalApp(BaseApp):
    def __init__(self, config):
        super().__init__(config)
        self.epoch = config.epoch

        image = mi.render(self.scene, spp=128)
        fig, ax = plt.subplots()
        ax.imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax.axis("off")
        ax.set_title(f"Path-traced image")
        plt.show()

    def train(self):
        self.integrator.set_config(self.config.grid.vmf_axis_encoding)

        # draw reference image
        self.integrator.set_train(False)
        self.integrator.set_path_trace(True)
        image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator)
        self.integrator.set_path_trace(False)
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

    def _render_epoch(self, epoch, image):
        if not self.should_render(epoch):
            return

        # Raw cache at x2: bsdf_w * VAPL(x2), no direct(x1).
        self.integrator.set_train(False)
        self.integrator.set_cache_only(True)
        image_x2 = np.array(
            mi.render(self.scene, spp=self.config.spp, integrator=self.integrator),
            dtype=np.float32,
        )
        self.integrator.set_cache_only(False)
        self.integrator.set_train(True)

        with torch.no_grad():
            gaussians_list, vmfs_list = self.grid.get_gaussians_for_debug_render()
            h, w = image.shape[0], image.shape[1]

            gaussians_list = gaussians_list[:self.config.grid.n_levels]
            vmfs_list = vmfs_list[:self.config.grid.n_levels]
            n_levels = len(gaussians_list)
            total_plots = n_levels + 2  # +1 for cache-at-x2 panel

            fig, axs = plt.subplots(1, total_plots, figsize=(6 * total_plots, 6), squeeze=False)
            axs = axs[0]

            axs[0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
            axs[0].axis("off")
            axs[0].set_title(f"cache at x1 - epoch:{epoch}")

            axs[1].imshow(np.clip(image_x2 ** (1.0 / 2.2), 0, 1))
            axs[1].axis("off")
            axs[1].set_title(f"cache at x2 (indirect only)")

            for level, (gaussians, vmfs) in enumerate(zip(gaussians_list, vmfs_list)):
                mean = gaussians[:, :3]
                variance = gaussians[:, 3]
                amplitude = vmfs[:, 4:7]
                axis = vmfs[:, 1:4]

                axs[level + 2].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                self.debug_vapl_render(self.scene, mean, variance, amplitude, axis, h, w, axs[level + 2])
                axs[level + 2].axis("off")
                axs[level + 2].set_title(f"level {level}")

            plt.tight_layout()
            plt.show()
