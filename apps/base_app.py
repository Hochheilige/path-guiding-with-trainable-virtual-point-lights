import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt
import torch

from integrator import RHSIntegrator
from grids import vapl_grid_base
from utils.scene import world_to_ndc, ndc_to_pixel


class BaseApp:
    def __init__(self, config):
        self.config = config
        self._load_scene()
        self._create_grid()
        self._create_integrator()

    def _load_scene(self):
        config = self.config
        if config.scene == "cornell box":
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

    def _create_grid(self):
        self.grid = vapl_grid_base.create_vapl_grid(
            self.config, self.scene.bbox().min, self.scene.bbox().max
        )

    def _create_integrator(self):
        self.integrator = RHSIntegrator(
            self.grid, True,
            loss_name=self.config.loss, indirect_only=self.config.indirect_only
        )
        self.integrator.set_depth(self.config.depth)

    def get_loss(self):
        return min(self.integrator.losses)

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

    def debug_vapl_render(self, scene, pos, variance, amplitude, axis, h, w, ax, show_directions=False):
        p = pos.cpu().detach().numpy()
        amplitude = amplitude.cpu().detach().numpy()
        variance = variance.cpu().detach().numpy().flatten()
        axis = axis.cpu().detach().numpy()

        # Filter by amplitude — keep only VPLs with non-negligible contribution
        # luminance = 0.2126 * amplitude[:, 0] + 0.7152 * amplitude[:, 1] + 0.0722 * amplitude[:, 2]
        # if luminance.size == 0 or luminance.max() == 0:
        #     return
        # threshold = luminance.max() * 0.01
        # mask = luminance > threshold
        # p = p[mask]
        # amplitude = amplitude[mask]
        # variance = variance[mask]
        # axis = axis[mask]

        if p.shape[0] == 0:
            return

        means_ndc = world_to_ndc(scene, p)
        means_pix = ndc_to_pixel(means_ndc, h, w)

        px = np.array(means_pix.x)
        py = np.array(means_pix.y)
        depth = np.array(means_ndc.z)

        # Filter: within image bounds and in front of camera
        visible = (px >= 0) & (px < w) & (py >= 0) & (py < h) & (depth > 0)

        px, py = px[visible], py[visible]
        depth = depth[visible]
        amplitude = amplitude[visible]
        variance = variance[visible]

        if px.shape[0] == 0:
            return

        # Sort by depth (far to near) so close VPLs draw on top
        order = np.argsort(-depth)
        px, py, depth = px[order], py[order], depth[order]
        amplitude = amplitude[order]
        variance = variance[order]

        # Depth-based size: closer = larger (inverse depth, scaled to [10, 80])
        inv_depth = 1.0 / (depth + 1e-8)
        inv_depth_norm = (inv_depth - inv_depth.min()) / (inv_depth.max() - inv_depth.min() + 1e-8)
        point_sizes = 10 + inv_depth_norm * 70

        # Depth-based alpha: closer = more opaque
        alpha = 0.3 + 0.7 * inv_depth_norm

        colors = amplitude / (amplitude.max() + 1e-8)
        # Add alpha channel
        colors = np.column_stack([colors, alpha])

        ax.scatter(px, py, c=colors, s=point_sizes, marker='o')

        if show_directions:
            axis = axis[visible][order]
            p = p[visible][order]
            axis_ndc = world_to_ndc(scene, p + axis * 0.05)
            axis_pix = ndc_to_pixel(axis_ndc, h, w)

            dx = np.array(axis_pix.x) - px
            dy = np.array(axis_pix.y) - py

            # Normalize arrow lengths to fixed pixel size for readability
            length = np.sqrt(dx**2 + dy**2) + 1e-8
            arrow_len = 8.0
            dx = dx / length * arrow_len
            dy = dy / length * arrow_len

            ax.quiver(px, py, dx, dy, angles='xy', scale_units='xy', scale=1, color=colors, width=0.003, headwidth=3)
