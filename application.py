import mitsuba as mi
mi.set_variant("cuda_ad_rgb")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wandb
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import time

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
            print(scene_dict)
            self.scene : mi.Scene = mi.load_dict(scene_dict)
        else:
            self.scene : mi.Scene = mi.load_file(config.scene)

        self.grid_vpl = vapl_grid_base.create_vapl_grid(config, self.scene.bbox().min, self.scene.bbox().max)
        config.grid.layout = "nrc"
        self.grid_nrc = vapl_grid_base.create_vapl_grid(config, self.scene.bbox().min, self.scene.bbox().max)
        self.loss_function = Loss(relativeL2)
        self.integrator_nrc = RHSIntegrator(self.grid_nrc, self.loss_function, True, True)
        self.integrator_vpl = RHSIntegrator(self.grid_vpl, self.loss_function, True)

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

        self.render_frames = []
        self.render_frames_debug = []

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

        total_time = 0
        for epoch in range(self.epoch):
            self.integrator_vpl.epoch = epoch
            start_time = time.perf_counter()
            image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator_vpl)
            end_time = time.perf_counter()
            total_time += end_time - start_time
            self.render(epoch, image)

        print("avg_time: ", total_time/self.epoch)
        print("epoch: ", self.epoch)

        total_time = 0
        #self.epoch += 4000
        for epoch in range(self.epoch):
            self.integrator_nrc.epoch = epoch
            start_time = time.perf_counter()
            image = mi.render(self.scene, spp=self.config.spp, integrator=self.integrator_nrc)
            end_time = time.perf_counter()
            total_time += end_time - start_time
            self.render(epoch, image)

        print("avg_time: ", total_time/self.epoch)
        print("epoch: ", self.epoch)

        if self.config.mode == "wandb":
            wandb.finish()

        imageio.mimsave("vapl_training.gif", self.render_frames, fps=5)
        imageio.mimsave("vapl_training_debug.gif", self.render_frames_debug, fps=5)

    def render_trained(self, spp):
        self.integrator_vpl.set_train(False)
        start_time = time.perf_counter()
        image_with_vpl_cache = mi.render(self.scene, spp=spp, integrator=self.integrator_vpl)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("VPL+PT time: ", total_time)


        self.integrator_nrc.set_train(False)
        start_time = time.perf_counter()
        image_with_nrc_cache = mi.render(self.scene, spp=spp, integrator=self.integrator_nrc)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("NRC+PT time: ", total_time)

        self.integrator_vpl.set_regular_pt()
        start_time = time.perf_counter()
        image_no_cache = mi.render(self.scene, spp=spp, integrator=self.integrator_vpl)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("PT time: ", total_time)

        gt = mi.render(self.scene, spp=1024)

        # Все изображения и соответствующие заголовки
        images = [
            (image_with_vpl_cache, "Path-traced image with VPL cache"),
            (image_with_nrc_cache, "Path-traced image with NRC cache"),
            (image_no_cache, "Path-traced image without cache"),
            (gt, "Reference")
        ]

        # Создание фигуры
        fig, ax = plt.subplots(len(images), 4, figsize=(15, 5 * len(images)))

        # Координаты областей зума
        regions = [
            {"x_start": 220, "x_end": 250, "y_start": 160, "y_end": 220},  # blue
            {"x_start": 70, "x_end": 100, "y_start": 87, "y_end": 169},  # red
            {"x_start": 130, "x_end": 190, "y_start": 65, "y_end": 95},  # green
        ]
        reg_colors = ["blue", "red", "green"]

        # Итерация по изображениям
        for i, (image, title) in enumerate(images):
            # Полное изображение
            ax[i, 0].imshow(np.clip(image ** (1.0 / 2.2), 0, 1))
            ax[i, 0].axis("off")
            ax[i, 0].set_title(title)

            # Добавление рамок для зум-областей
            for region, color in zip(regions, reg_colors):
                rect = plt.Rectangle(
                    (region["y_start"], region["x_start"]),
                    region["y_end"] - region["y_start"],
                    region["x_end"] - region["x_start"],
                    linewidth=2, edgecolor=color, facecolor='none'
                )
                ax[i, 0].add_patch(rect)

            # Зум-области
            for j, region in enumerate(regions):
                zoom_image = image[region["x_start"]:region["x_end"], region["y_start"]:region["y_end"]]
                ax[i, j + 1].imshow(np.clip(zoom_image ** (1.0 / 2.2), 0, 1), origin='upper')
                ax[i, j + 1].axis("off")
                ax[i, j + 1].set_title(f"Zoom {j + 1} ({title})")

                # Рамка для зум-области
                rect_zoom = patches.Rectangle(
                    (-0.5, -0.5),  # Верхний левый угол
                    zoom_image.shape[1],  # Ширина
                    zoom_image.shape[0],  # Высота
                    linewidth=10,
                    edgecolor=reg_colors[j],
                    facecolor='none'
                )
                ax[i, j + 1].add_patch(rect_zoom)

        plt.tight_layout()
        plt.show()

    def render(self, epoch, image):
        if (self.config.mode == "wandb" or self.config.mode == "sweep"):
            wandb.log({"loss": self.integrator.losses[-1].item(), "epoch": epoch})

        if (self.should_render(epoch)):
            with torch.no_grad():
                gaussians, vmfs = self.grid_vpl.get_gaussians_for_debug_render()

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

                # debug gif for presentation
                frame = np.clip(image ** (1.0 / 2.2), 0, 1)
                frame = (frame * 255).astype(np.uint8)
                self.render_frames.append(frame)

                canvas = FigureCanvas(fig)
                canvas.draw()
                width, height = canvas.get_width_height()
                rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
                bbox = ax[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                x0, y0, x1, y1 = bbox.extents
                x0, y0, x1, y1 = map(int, [x0 * fig.dpi, y0 * fig.dpi, x1 * fig.dpi, y1 * fig.dpi])
                frame_ax1 = rgba[y0:y1, x0:x1, :3]
                self.render_frames_debug.append(frame_ax1)

                plt.close(fig)

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

        point_sizes = 25 * variance

        axis_nds = world_to_ndc(scene, axis.cpu().detach().numpy())
        axis_pix = ndc_to_pixel(axis_nds, h, w)

        dx = axis_pix.x - means_pix.x
        dy = axis_pix.y - means_pix.y

        ax.scatter(means_pix.x, means_pix.y, c=colors, cmap='coolwarm', marker='o', s=point_sizes)
        # TODO: figure out how to render arrows more correct
        #ax.quiver(means_pix.x, means_pix.y, dx, dy, angles='uv', color=colors, scale=1, scale_units='xy')
