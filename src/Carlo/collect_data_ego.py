import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
Path("../csvfiles_ego").mkdir(parents=True, exist_ok=True)  # creates new folder

from utils import *
from scenarios import *
from fourway_intersection import build_world

from random import choice
import numpy as np
import pandas as pd

from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

from tqdm import tqdm
from multiprocessing import cpu_count, Pool
# from joblib import Parallel, delayed
import istarmap  # import to apply patch


def get_inputs(id, ego_dirs, ts_total_min, ts_total_max):
    init_dir, final_dir = choice(ego_dirs)
    ts_total  = np.random.randint(ts_total_min, ts_total_max)

    # print((init_dir, final_dir, pos_noise, ang_noise, ts_total))
    return (id, init_dir, final_dir, ts_total)


def loop(id, init_dir, final_dir, ts_total):
    # Build the fourway intersection world
    dt = 0.1
    w = build_world(dt)

    c1 = spawn_car(dt, ts_total, init_dir, final_dir, pos_path_noise=0.01)
    c1.set_control(0, 0)
    w.add(c1)

    Rows = []

    for ts in range(ts_total):
        if w.collision_exists(c1): return w.close()  # return without recording

        pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)

        u_throttle, _ = find_nearest(u_throttle_allowed_values, u_throttle)
        u_steering, _ = find_nearest(u_steering_allowed_values, u_steering)

        c1.set_control(u_steering, u_throttle)
    
        w.tick()
        if render: w.render(); time.sleep(dt/20)

        Rows.append([ts, c1.x, c1.y, c1.xp, c1.yp, c1.heading, init_dir, final_dir, u_steering, u_throttle])


    if close_to(final_position(final_dir), [c1.x, c1.y]):
        Data = empty_df()
        Data = Data.append(pd.DataFrame(Rows, columns=Data.columns), ignore_index=True)
        dump_csv(Data, id, cartype="ego")

    return w.close()



if __name__ == "__main__":

    ### Params ###
    render = False
    parallel = True
    u_throttle_allowed_values = np.linspace(-25, +25, 3)
    u_steering_allowed_values = np.linspace(-5, +5, 200)
    num_of_runs = 10000
    ts_total_min = 20
    ts_total_max = 300
    ##############


    ego_dirs = [("south", "west")]
    iterable = [get_inputs(id, ego_dirs, ts_total_min, ts_total_max) for id in tqdm(range(num_of_runs), desc="Building scenarios")]

    if parallel:
        if render: raise Exception("Cannot render graphics while parallelized.")

        num_cores = cpu_count()-1
        with Pool(num_cores) as pool:
            for _ in tqdm(pool.istarmap(loop, iterable), total=len(iterable), desc="Playing scenarios"):
                pass

    else:
        for items in tqdm(iterable, desc="Playing scenarios"):
            loop(*items)