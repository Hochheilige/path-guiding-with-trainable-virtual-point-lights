import gc
import torch
import drjit as dr
import wandb

from integrator import RHSIntegrator
from .wandb_app import WandbApp


class SweepApp:
    def __init__(self, config):
        self.config = config
        self._app = None

    def sweep(self):
        wandb.init(project="vapls-parameters-encodings-search")
        self._recreate()
        self._train()

    def _recreate(self):
        if self._app is not None:
            for attr in ('grid', 'integrator', 'scene'):
                if hasattr(self._app, attr):
                    delattr(self._app, attr)
            self._app = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        dr.kernel_history_clear()
        dr.flush_malloc_cache()
        dr.flush_kernel_cache()

        self.config.sweep_config = wandb.config

        # Propagate sweep encoding parameters into config.grid so create_vapl_grid sees them
        grid_encoding_keys = [
            'gaussian_mean_encoding', 'gaussian_variance_encoding',
            'vmf_sharpness_encoding', 'vmf_axis_encoding', 'vmf_amplitude_encoding',
        ]
        for key in grid_encoding_keys:
            if key in wandb.config:
                self.config.grid[key] = wandb.config[key]

        self._app = WandbApp(self.config)

        self._app.integrator = RHSIntegrator(
            self._app.grid, True,
            loss_name=self.config.loss, indirect_only=self.config.indirect_only
        )
        self._app.integrator.set_depth(self.config.depth)

        axis_enc = wandb.config.get('vmf_axis_encoding', self.config.grid.vmf_axis_encoding)
        self._app.integrator.set_config(axis_enc)

        self._app.epoch = self.config.sweep_config.epoch

    def _train(self):
        self._app.train(_skip_wandb_init=True)

    def get_loss(self):
        return self._app.get_loss()
