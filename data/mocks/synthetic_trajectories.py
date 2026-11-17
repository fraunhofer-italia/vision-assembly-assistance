import numpy as np
import copy
import random

TOOLS = ["NONE", "ScrewdriverL", "ScrewdriverS"]

def generate_expert_trajectory(
        n_points = 50,
        tool = "ScrewdriverL"
):
    """
    Generates a smooth expert trajectory (e.g. arc /  line)
    
    :param n_points: Description
    :param tool: Description
    """
    traj = []
    for i in range(n_points):
        t = i*0.05
        x = 0.2 + 0.6 * (i / n_points)
        y = 0.3 + 0.2 * np.sin(2 * np.pi * i / n_points)

        traj.append({
            "x" : x,
            "y" : y,
            "t" : t,
            "tool" : tool
        })

    return traj

def add_spatial_noise(traj, sigma=0.01):
    noisy = copy.deepcopy(traj)
    for p in noisy:
        p["x"] += np.random.normal(0, sigma)
        p["y"] += np.random.normal(0, sigma)
    return noisy

def add_temporal_jitter(traj, max_shift=0.02):
    noisy = copy.deepcopy(traj)
    for p in noisy:
        p["t"] += random.uniform(-max_shift, max_shift)
    return noisy

def add_tool_noise(traj, flip_prob=0.1):
    noisy = copy.deepcopy(traj)
    for p in noisy:
        if random.random() < flip_prob:
            p["tool"] = random.choice(TOOLS)
    return noisy
