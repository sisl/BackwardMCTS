from utils import *
from simple_pid import PID

from world import World
from agents import Car, RectangleBuilding, Pedestrian, Painting
from geometry import Point

def populate_rival_directions():
    dirs = ['south', 'north', 'east', 'west']
    return [(x,y) for x in dirs for y in dirs if x != y]

def init_angle(dir):
    table = {'south': 0.5*np.pi,
             'north': 1.5*np.pi,
             'east':  np.pi,
             'west':  0.0}
    return table[dir]

def init_position(dir):
    table = {'south': [63,50],
             'north': [57,70],
             'east':  [70,63],
             'west':  [50,57]}
    return np.array(table[dir])

def final_position(dir):
    table = {'south': [57,50],
             'north': [63,70],
             'east':  [70,57],
             'west':  [50,63]}
    return np.array(table[dir])

def mid_point_position(init, final):

    mid_point = [60,60]
    val, idx = find_nearest(init_position(init), 60)
    mid_point[idx] = val

    val, idx = find_nearest(final_position(final), 60)
    mid_point[idx] = val

    return mid_point


def final_angle(dir):
    table = {'south': 1.5*np.pi,
             'north': 0.5*np.pi,
             'east':  0.0,
             'west':  np.pi}
    return table[dir]

def ref_position(dir):
    table = {'south': 1,
             'north': 1,
             'east':  0,
             'west':  0}
    return table[dir]


def init_ref_sign(dir):
    table = {'south': +1,
             'north': -1,
             'east':  -1,
             'west':  +1}
    return table[dir]


def final_ref_sign(dir):
    table = {'south': -1,
             'north': +1,
             'east':  +1,
             'west':  -1}
    return table[dir]

def get_position_path(initial, final, timesteps, noise_std=0.0):
    mid_point = mid_point_position(initial, final)
    first_half = np.linspace(init_position(initial), mid_point, timesteps//2)
    second_half = np.linspace(mid_point, final_position(final), timesteps//2)

    result = np.vstack([first_half, second_half])
    return result + np.random.normal(scale=noise_std, size=result.shape)

def get_angular_path(initial, final, timesteps, noise_std=0.0):
    mid = np.linspace(initial, final, timesteps//3).reshape(-1,1)
    z0 = np.ones_like(mid) * initial
    z1 = np.ones_like(mid) * final

    result = np.vstack([z0, mid, z1])
    return result + np.random.normal(scale=noise_std, size=result.shape)

def get_pos_diff(real, target, idx):
    dx = target-real
    # return np.sign(dx) * np.linalg.norm(dx)
    # return np.linalg.norm(dx)
    return dx[idx]
    # return np.sum(dx)


def get_ang_diff(real, target):
    diff = target-real
    # return (diff + np.pi) % 2*np.pi - np.pi
    if diff > np.pi: return diff - 2*np.pi
    elif diff < -np.pi: return diff + 2*np.pi
    else: return diff


def get_controls(car, dt):
    ts = car.ts_now
    # x = myround(car.x, split_into=4)
    # y = myround(car.y, split_into=4)

    if ts < car.ts_total//2:
        idx=ref_position(car.init_dir)
        sgn=init_ref_sign(car.init_dir)
    else: 
        idx=ref_position(car.final_dir)
        sgn=final_ref_sign(car.final_dir)

    # Get throttle value.
    pos_diff = sgn * get_pos_diff(car.pos_path[ts], np.array([car.x, car.y]), idx=idx)
    u_throttle = car.pos_controller(pos_diff, dt=dt)

    # Get steering value.
    desired_heading = np.arctan2(car.pos_path[ts+1][1] - car.y , car.pos_path[ts+1][0] - car.x)
    ang_diff = get_ang_diff(desired_heading, car.heading)  
    u_steering = car.ang_controller(ang_diff, dt=dt)

    if ts + 2 < car.pos_path.shape[0]: car.ts_now += 1
    return pos_diff, ang_diff, u_steering, u_throttle


def spawn_car(dt, timesteps, init='west', final='north', pos_path_noise=0.0):
    if (init, final) not in populate_rival_directions(): raise Exception("Invalid `init` of `final`.")

    car = Car(Point(*init_position(init)), init_angle(init), init_dir=init, final_dir=final, ts_total=timesteps)

    car.pos_path = get_position_path(initial=init, final=final, timesteps=timesteps, noise_std=pos_path_noise)

    car.pos_controller = PID(Kp=20.0, Ki=0.1, Kd=20.0, sample_time=dt, setpoint=0)
    car.ang_controller = PID(Kp=1.0, Ki=0.001, Kd=0.01, sample_time=dt, setpoint=0)

    return car