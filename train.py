import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from application import Application
from vapl_config import config

config.mode = "sweep"


def train():
    app = Application(config)
    app.sweep()


if __name__ == "__main__":
    train()
