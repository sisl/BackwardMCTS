#!/usr/bin/env python3

import yaml
from pathlib import Path


with Path(__file__).with_name('run_config.yaml').open('r') as Y:
    with Path(__file__).with_name('parallel_runs.cmd').open('w') as F:

        conf = yaml.safe_load(Y)

        for T in conf['timesteps_range']:
            for SZ in conf['gridsize_range']:
                for TO in conf['t_and_o_probs_range']:
                    for Z in conf['z_val_range']:

                        F.write(f"julia --timesteps {T} --gridsize {SZ} --t_and_o_prob {TO} --z_val {Z} \n")
      
        F.close()
    Y.close()