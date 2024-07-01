# This script replicates the results from Figure 1
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/teaser/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

mesh = "rossignol"
# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)
# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

ns = [ 20, 55 ]
    
for i, n in enumerate(ns):

    print(f"Running for n = {n}")

    # Set up a grid
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    # Marching cubes
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

    # ndc
    # V_ndc, F_ndc = V_mc, F_mc # Just so it doesn't crash on computers without ndc

    # Reach for the Spheres
    V0, F0 = gpy.icosphere(2)
    V_rfts, F_rfts = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)

    # Reach for the Arcs
    V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True, fine_tune_iters=50)

    write_path = results_path + mesh +  f"/{n}/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    gpy.write_mesh( write_path + "marching_cubes.obj", V_mc, F_mc)
    # gpy.write_mesh( write_path + "ndc.obj", V_ndc, F_ndc)
    gpy.write_mesh( write_path + "rfts.obj", V_rfts, F_rfts)
    gpy.write_mesh( write_path + "rfta.obj", V_rfta, F_rfta)
    gpy.write_mesh( write_path + "ground_truth.obj", V_gt, F_gt)
    