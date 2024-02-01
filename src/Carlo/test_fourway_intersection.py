from utils import *
from scenarios import *
from fourway_intersection import build_world

import numpy as np
from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point
import time

def test(u_throttle_allowed_values, u_steering_allowed_values):

    # Build the fourway intersection world
    dt = 0.1
    w = build_world(dt)
    
    # Add pedestrians to critical coordinates
    for Y in [50,55,60,65,70]:
        for X in [50,55,60,65,70]:
            p1 = Pedestrian(Point(X,Y), 0.0)
            w.add(p1)

    ts_total = 30

    for (init, final) in [("east", "north")]:   # populate_rival_directions():

        c1 = spawn_car(dt, timesteps=ts_total, init=init, final=final, pos_path_noise=0.01)
        c1.set_control(0, 0)
        w.add(c1)

        c2 = Car(Point(100,60), np.pi, 'blue')
        c2.velocity = Point(3.0,0) # We can also specify an initial velocity just like this.
        w.add(c2)

        w.render()


        for ts in range(ts_total):
            if w.collision_exists(): # we can check if there is any collision.
                print('Collision exists somewhere...')

            pos_diff, ang_diff, u_steering, u_throttle = get_controls(c1, dt)

            u_steering, _ = find_nearest(u_steering_allowed_values, u_steering)
            u_throttle, _ = find_nearest(u_throttle_allowed_values, u_throttle)

            c1.set_control(u_steering, u_throttle)
            
            w.tick() # This ticks the world for one time step (dt second)
            w.render()
            time.sleep(dt/10) # Let's watch it 4x

            print(f"Timestep: {ts}, Pos_Diff: {pos_diff} and u_th: {u_throttle}  |  Ang_Diff: {ang_diff} and u_st: {u_steering}")

        c1.set_control(0, -np.Inf)
        print((init, final))

        import ipdb; ipdb.set_trace()

if __name__ == "__main__":

    ### Params ###
    u_throttle_allowed_values = np.linspace(-25, +25, 3)
    u_steering_allowed_values = np.linspace(-5, +5, 200)
    ##############

    test(u_throttle_allowed_values, u_steering_allowed_values)