import os
import time 
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from omegaconf import OmegaConf

import pyflex
from softgym.envs.bimanual_env import BimanualEnv
from softgym.envs.bimanual_tshirt import BimanualTshirtEnv

class EnvRollout(object):
    def __init__(self, args):
        self.args = args

        if 'towel' in args.cloth_type:
            self.env = BimanualEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless,
                    shape='default' if 'square' in args.cloth_type else 'rect')
        elif args.cloth_type == 'tshirt':
            self.env = BimanualTshirtEnv(use_depth=True,
                    use_cached_states=False,
                    horizon=1,
                    action_repeat=1,
                    headless=args.headless)