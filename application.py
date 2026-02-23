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
        self.integrator = RHSIntegrator(self.grid, self.loss_function, True,
                                        drjit_loss_name=config.loss)
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
        self.integrator = RHSIntegrator(self.grid, self.loss_function, True,
                                        drjit_loss_name=self.config.loss)
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

        # # cornell box with specular sphere
        # scene_dict = mi.cornell_box()
        # scene_dict['sphere'] = {
        #        'type': 'sphere',
        #        'radius': 0.4,
        #        'center': [0, 0.2, 0],
        #        'bsdf': {
        #            'type': 'roughconductor',
        #            'distribution': 'ggx',
        #            'alpha_u': 0.5,
        #            'alpha_v': 0.1
        #        },
        # }
        # scene_dict['sensor']['to_world'] = mi.ScalarTransform4f.look_at(
        #     origin=[0.8, 0.8, 0.8],   # left side
        #     target=[0.0, 0.0, 0.0],    # look at center
        #     up=[0, 1, 0]
        # )
        # self.scene = mi.load_dict(scene_dict)

        self.integrator.set_train(False)
        self.integrator.set_vapl_ratio(0.0)
        image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator)
        fig, ax = plt.subplots()
        ax.imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax.axis("off")
        ax.set_title(f"PT")
        plt.show()

        self.integrator.set_vapl_ratio(0.95)
        image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator)
        fig, ax = plt.subplots()
        ax.imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
        ax.axis("off")
        ax.set_title(f"PT + VPL")
        plt.show()


        if self.config.mode == "wandb":
            wandb.finish()

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

    def render(self, epoch, image):
        if self.config.mode in ["wandb", "sweep"]:
            last_loss = self.integrator.losses[-1]
            loss_val = last_loss.item() if hasattr(last_loss, 'item') else float(last_loss)
            wandb.log({"loss": loss_val, "epoch": epoch})

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

                axs[level + 1].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
                self.debug_vapl_render(self.scene, mean, variance, amplitude, axis, h, w, axs[level + 1])
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

