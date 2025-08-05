import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

import wandb
import vapl_config
import vapl_grid
from integrator import *

if vapl_config.config.scene == "cornell box":
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
    scene = mi.load_dict(scene_dict)
else:
    scene = mi.load_file(vapl_config.config.scene)

def train():
    wandb.init()
    wandb_config = wandb.config
    config = vapl_config.config
    config.mode = "sweep"
    config.sweep_config = wandb_config
    grid = vapl_grid.vapl_grid_base.create_vapl_grid(config, scene.bbox().min, scene.bbox().max)
    loss_fn = Loss(relativeL2_luminance_tiny_cuda_nn)
    integrator = RHSIntegrator(grid, loss_fn, True)
    integrator.set_depth(config.depth)
    integrator.set_config(wandb_config.vmf_axis_encoding)

    # store GT image
    integrator.set_train(False)
    mi.render(scene, spp=config.spp, integrator=integrator)
    integrator.set_train(True)

    for epoch in range(wandb_config.epoch):
        integrator.epoch = epoch
        mi.render(scene, spp=config.spp, integrator=integrator)
        wandb.log({"loss": integrator.losses[-1].item(), "epoch": epoch})

if __name__ == "__main__":
    train()
