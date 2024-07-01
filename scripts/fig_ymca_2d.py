# This script replicates the 2D results from Figure 6
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/ymca_2d/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

meshes = ["letter_R", "letter_F", "letter_T", "letter_A"]
for mesh in meshes:
    print(f"Doing {mesh}")

    # Set up gt
    filename = "data/" + mesh + ".png"
    poly_list = gpy.png2poly(filename)
    V_gt = None
    F_gt = None
    for poly in poly_list:
        nv = 0 if V_gt is None else V_gt.shape[0]
        pV = poly[::5,:]
        V_gt = pV if V_gt is None else np.concatenate((V_gt, pV), axis=0)
        F_gt = gpy.edge_indices(pV.shape[0],closed=True) if F_gt is None else \
            np.concatenate((F_gt, nv+gpy.edge_indices(pV.shape[0],closed=True)),
                axis=0)
    V_gt = gpy.normalize_points(V_gt)

    # Create and abstract SDF function that is the only connection to the shape
    sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

    n = 85

    # Set up a grid
    gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten())).T
    S = sdf(U)

    # Marching cubes
    print("  MC")
    V_mc, F_mc = gpy.marching_squares(S, U, n+1, n+1)

    # Reach for the Arcs
    print("  RFTA")
    V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True,
        fine_tune_iters=50, max_points_per_sphere=3)

    # Reach for the Spheres
    print("  RFTS")
    V0, F0 = gpy.regular_circle_polyline(12)
    V_rfts, F_rfts = gpy.reach_for_the_spheres(U, None, V0, F0, S=S,
        max_iter=100, verbose=False)


    write_path = results_path + mesh +  "/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    utility.write_mesh( write_path + "marching_cubes.npy", V_mc, F_mc)
    utility.write_mesh( write_path + "rfts.npy", V_rfts, F_rfts)
    utility.write_mesh( write_path + "rfta.npy", V_rfta, F_rfta)
    utility.write_mesh( write_path + "ground_truth.npy", V_gt, F_gt)


    