#!/usr/bin/env python3

# Run this script.
# Then bash `parallel < parallel_runs.cmd`.

import yaml
from itertools import product
from pathlib import Path


with Path(__file__).with_name('run_config.yaml').open('r') as Y:
    with Path(__file__).with_name('parallel_runs.cmd').open('w') as F:

        conf = yaml.safe_load(Y)
        combinations = [conf['timesteps_range'], conf['gridsize_range'], conf['t_and_o_probs_range'], conf['z_threshold_range']]
        ON, BN, SN = conf['obs_N'], conf['belief_N'], conf['savename']
        
        for (T, SZ, TO, Z) in product(*combinations):
            F.write(f"julia run_benchmark.jl --timesteps {T} --gridsize {SZ} --t_and_o_prob {TO} --z_threshold {Z}  --obs_N {ON} --belief_N {BN} --savename {SN} \n")
