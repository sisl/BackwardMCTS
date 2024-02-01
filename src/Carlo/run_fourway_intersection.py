from utils import *
from scenarios import *
from fourway_intersection import build_world

import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

# Build the fourway intersection world
dt = 0.1
w = build_world(dt)

render = True
ts_total = 177
(init, final) = ('east', 'south')

c1 = spawn_car(dt, timesteps=ts_total, init=init, final=final, pos_path_noise=0.2)
c1.set_control(0, 0)
w.add(c1)

c2 = Car(Point(100,60), np.pi, 'blue')
c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
w.add(c2)



for ts in range(400):
    # if w.collision_exists(): # we can check if there is any collision.
    #     print('Collision exists somewhere...')

    pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)
    c1.set_control(u_steering, u_throttle)
    
    w.tick() # This ticks the world for one time step (dt second)
    if render:
        w.render()
        time.sleep(dt/10) # Let's watch it 10x

    print(f"Timestep: {ts}, Pos_Diff: {pos_diff} and u_th: {u_throttle}  |  Ang_Diff: {ang_diff} and u_st: {u_steering}")

print((init, final))

