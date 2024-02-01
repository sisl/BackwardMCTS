import pandas as pd
import numpy as np
import time

def clamp(num, min_value, max_value):
   return max(min(num, max_value), min_value)
   
def empty_df():
    df_values = {'Timestep' : [],
                'X_Position': [],
                'Y_Position': [],
                'X_Velocity': [],
                'Y_Velocity': [],
                'Heading'   : [],
                'StartDir'  : [],
                'FinalDir'  : [],
                'U_Steering': [],
                'U_Throttle': [],
                }

    return pd.DataFrame(df_values)

def dump_csv(obj, id, cartype):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    savename = f"../csvfiles_{cartype}/{timestamp}_{id}.csv"
    return obj.to_csv(savename, index=False)

def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(a - a0).argmin()
    return a.flat[idx], idx

def extrema(arr):
    a = np.array(arr)
    return a.min(), a.mean(), a.max()

def myround(number, split_into=2):
    "Rounds `number` to the nearest 0.5 when `split_into` == 2"
    dv = 1/split_into
    return round(number * dv) / dv

def close_to(a1, a2):
    return np.linalg.norm(np.array(a1) - np.array(a2)) < 5  # [meters]