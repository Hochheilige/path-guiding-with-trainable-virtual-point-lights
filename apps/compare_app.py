import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import numpy as np
import matplotlib.pyplot as plt
import flip_evaluator
import wandb

from integrator import RHSIntegrator
from grids import vapl_grid_base
from grids.nrc_drjit import nrc_model_drjit

from .base_app import BaseApp


def _gamma(img):
    return np.clip(np.array(img, dtype=np.float32) ** (1.0 / 2.2), 0.0, 1.0)

def _show(ax, img, title, gamma=True, cmap=None):
    disp = _gamma(img) if gamma else np.clip(np.array(img), 0.0, 1.0)
    ax.imshow(disp, cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis('off')

def _flip(test, reference):
    """HDR-FLIP via flip_evaluator. Returns (map H×W×3, mean scalar)."""
    flip_map, mean_val, _ = flip_evaluator.evaluate(
        np.array(test,      dtype=np.float32),
        np.array(reference, dtype=np.float32),
        'HDR',
    )
    return flip_map, float(mean_val)


class CompareApp(BaseApp):
    """Trains NRC and VAPL drjit grid side-by-side.

    Every display epoch renders one figure:

              col 0                col 1               col 2
    row 0:  Reference (high spp)  GT (1 spp)
    row 1:  NRC  cache output     NRC  inference       NRC  FLIP
    row 2:  VAPL cache output     VAPL inference       VAPL FLIP

    config.grid.layout sets the VAPL grid type ("drjit" or "drjit-mlp").
    If layout is "nrc-drjit" it is overridden to "drjit" for the VAPL model.
    NRC is always nrc_model_drjit regardless of layout.

    Set config.mode = "wandb" to enable WandB logging.
    """

    def __init__(self, config):
        super().__init__(config)

        ref_spp = getattr(config, 'ref_spp', 512)
        print(f"Rendering reference ({ref_spp} spp)...")
        self.reference = np.array(mi.render(self.scene, spp=ref_spp), dtype=np.float32)
        print("Reference done.")

        if config.mode == "wandb":
            wandb.login()

    def _create_grid(self):
        bb_min = self.scene.bbox().min
        bb_max = self.scene.bbox().max

        # VAPL grid — use "drjit" layout even if config says "nrc-drjit"
        original_layout = self.config.grid.layout
        if original_layout == "nrc-drjit":
            self.config.grid['layout'] = "drjit"
        self.vapl_model = vapl_grid_base.create_vapl_grid(self.config, bb_min, bb_max)
        self.config.grid['layout'] = original_layout
        self.grid = self.vapl_model  # base class compat

        self.nrc_model = nrc_model_drjit(self.config, bb_min=bb_min, bb_max=bb_max)

    def _create_integrator(self):
        self.vapl_integrator = RHSIntegrator(
            self.vapl_model, True,
            loss_name=self.config.loss, indirect_only=self.config.indirect_only,
        )
        self.vapl_integrator.set_depth(self.config.depth)

        self.nrc_integrator = RHSIntegrator(
            self.nrc_model, True,
            loss_name=self.config.loss, indirect_only=self.config.indirect_only,
            nrc_depth=self.config.nrc_suffix_depth,
        )
        self.nrc_integrator.set_depth(self.config.depth)

        self.integrator = self.vapl_integrator  # base class compat

    # -------------------------------------------------------------------------

    def train(self):
        use_wandb = self.config.mode == "wandb"
        if use_wandb:
            init_kwargs = dict(
                project="vapls-training",
                name=self.config.run_name,
                config=self.config,
            )
            if self.config.wandb_group != "default":
                init_kwargs["group"] = self.config.wandb_group
            wandb.init(**init_kwargs)
            wandb.define_metric("*", step_metric="epoch")

        try:
            self._train_loop(use_wandb)
        finally:
            if use_wandb:
                wandb.finish()

    def _train_loop(self, use_wandb):
        for epoch in range(self.config.epoch):

            # Shared 1-spp GT via path tracing
            self.nrc_integrator.set_train(False)
            self.nrc_integrator.set_path_trace(True)
            gt_img = np.array(
                mi.render(self.scene, spp=self.config.spp, integrator=self.nrc_integrator),
                dtype=np.float32,
            )
            self.nrc_integrator.set_path_trace(False)

            # NRC training
            self.nrc_integrator.set_train(True)
            self.nrc_integrator.epoch = epoch
            nrc_cache_img = np.array(
                mi.render(self.scene, spp=self.config.spp, integrator=self.nrc_integrator),
                dtype=np.float32,
            )

            # VAPL training
            self.vapl_integrator.set_train(True)
            self.vapl_integrator.epoch = epoch
            vapl_cache_img = np.array(
                mi.render(self.scene, spp=self.config.spp, integrator=self.vapl_integrator),
                dtype=np.float32,
            )

            nrc_loss  = float(self.nrc_integrator.losses[-1])  if self.nrc_integrator.losses  else float('nan')
            vapl_loss = float(self.vapl_integrator.losses[-1]) if self.vapl_integrator.losses else float('nan')

            if not self.should_render(epoch):
                if use_wandb:
                    wandb.log({"epoch": epoch, "nrc/loss": nrc_loss, "vapl/loss": vapl_loss})
                continue

            # NRC inference
            self.nrc_integrator.set_train(False)
            nrc_render = np.array(
                mi.render(self.scene, spp=self.config.spp, integrator=self.nrc_integrator),
                dtype=np.float32,
            )
            self.nrc_integrator.set_train(True)

            # VAPL inference
            self.vapl_integrator.set_train(False)
            vapl_render = np.array(
                mi.render(self.scene, spp=self.config.spp, integrator=self.vapl_integrator),
                dtype=np.float32,
            )
            self.vapl_integrator.set_train(True)

            nrc_flip,  nrc_flip_mean  = _flip(nrc_render,  self.reference)
            vapl_flip, vapl_flip_mean = _flip(vapl_render, self.reference)

            self._display(
                epoch, gt_img,
                nrc_cache_img,  nrc_render,  nrc_flip,  nrc_flip_mean,  nrc_loss,
                vapl_cache_img, vapl_render, vapl_flip, vapl_flip_mean, vapl_loss,
                use_wandb=use_wandb,
            )

    # -------------------------------------------------------------------------

    def _display(self, epoch, gt,
                 nrc_cache,  nrc_render,  nrc_flip,  nrc_flip_mean,  nrc_loss,
                 vapl_cache, vapl_render, vapl_flip, vapl_flip_mean, vapl_loss,
                 use_wandb=False):
        panel_in = 5
        dpi      = 150

        fig, axs = plt.subplots(3, 3, figsize=(panel_in * 3, panel_in * 3), dpi=dpi)
        fig.suptitle(
            f'Epoch {epoch}   |   '
            f'NRC  FLIP={nrc_flip_mean:.4f}  loss={nrc_loss:.4f}   |   '
            f'VAPL FLIP={vapl_flip_mean:.4f}  loss={vapl_loss:.4f}',
            fontsize=11, fontweight='bold',
        )

        ref_spp = getattr(self.config, 'ref_spp', 512)

        # row 0 — shared
        _show(axs[0, 0], self.reference, f'Reference ({ref_spp} spp)')
        _show(axs[0, 1], gt,             'GT (1 spp)')
        axs[0, 2].axis('off')

        # row 1 — NRC
        _show(axs[1, 0], nrc_cache,  'NRC cache (train output)')
        _show(axs[1, 1], nrc_render, 'NRC inference')
        _show(axs[1, 2], nrc_flip,   f'NRC FLIP={nrc_flip_mean:.4f}', gamma=False, cmap='magma')

        # row 2 — VAPL
        _show(axs[2, 0], vapl_cache,  'VAPL cache (train output)')
        _show(axs[2, 1], vapl_render, 'VAPL inference')
        _show(axs[2, 2], vapl_flip,   f'VAPL FLIP={vapl_flip_mean:.4f}', gamma=False, cmap='magma')

        plt.tight_layout()

        if use_wandb:
            wandb.log({
                "epoch":      epoch,
                "nrc/loss":   nrc_loss,
                "vapl/loss":  vapl_loss,
                "nrc/flip":   nrc_flip_mean,
                "vapl/flip":  vapl_flip_mean,
                "comparison": wandb.Image(fig),
            })

        plt.show()
        plt.close(fig)
