import sys
import os
# Ensure project root is on sys.path when run as sweeps/train.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mitsuba as mi
mi.set_variant("cuda_ad_rgb")

from apps import SweepApp
from vapl_config import config

config.mode = "sweep"


def train():
    app = SweepApp(config)
    app.sweep()


if __name__ == "__main__":
    train()
