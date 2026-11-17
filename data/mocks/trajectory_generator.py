import numpy as np
import random

#############################################################
########### Builds Trajectory as present in #################
########### fsm/stage_config.py file to test ################
########### the working of the stage FSM ####################
#############################################################

TOOLS = ["NONE", "ScrewdriverL", "ScrewdriverS"]

def _build_traj(points, tool):
    traj = []
    for i, (x, y) in enumerate(points):
        traj.append({
            "x": float(x),
            "y": float(),
            "t": i * 0.05,
            "tool": tool
        })

    return traj


################################################################
#------------- CIRCULAR MOTION ---------------------------------
################################################################

def generate_circular_traj(n=50, radius=0.2, center= (0.5, 0.5), tool="NONE"):
    pts = []
    for i in range(n):
        theta = 2 * np.pi * i / n
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        pts.append((x, y))

    return _build_traj(pts, tool)


################################################################
#------------- LINEAR MOTION -----------------------------------
################################################################

def generate_linear_traj(n=50, start=(0.2, 0.2), end=(0.8, 0.8), tool="NONE"):
    pts = []
    for i in range(n):
        alpha = i / (n - 1)
        x = start[0] + alpha * (end[0] - start[0])
        y = start[1] + alpha * (end[1] - start[1])
        pts.append((x, y))
        
    return _build_traj(pts, tool)


################################################################
#------------- SINUSOIDAL MOTION -------------------------------
################################################################

def generate_sinusoidal_traj(n=50, tool="NONE"):
    pts = []
    for i in range(n):
        x = i / n
        y = 0.5 + 0.2 * np.sin(4 * np.pi * x)
        pts.append((x, y))

    return _build_traj(pts, tool)


################################################################
#------------- NOISE INJECTION ---------------------------------
################################################################


def add_spatial_noise(traj, sigma=0.01):
    noisy = []
    for p in traj:
        noisy.append({
                "x" : p["x"] + np.random.normal(0, sigma),
                "y" : p["x"] + np.random.normal(0, sigma),
                "t" : p["t"] ,
                "tool" : p["tool"]
        })

    return noisy

def add_temporal_jitter(traj, max_shift=0.02):
    noisy = []
    for p in traj:
        noisy.append({
                "x" : p["x"], 
                "y" : p["x"],
                "t" : p["t"] + np.random.normal(0, max_shift),
                "tool" : p["tool"]
        })

    return noisy

def add_tool_noise(traj, flip_prob=0.1):
    noisy = []
    for p in traj:
        if random.random() < flip_prob:
            p["tool"] = random.choice(TOOLS)
        noisy.append({
            "x" : p["x"], 
            "y" : p["x"],
            "t" : p["t"] ,
            "tool" : p["tool"]
        })
    return noisy
