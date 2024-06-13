# This script replicates the 3D results from Figure 6
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 3523

results_path = "results/ymca/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

methods = ["RFTS", "MC", "RFTA"]
meshes = ["human_R", "human_F", "human_T", "human_A"]
for method in methods:
    for mesh in meshes:
        print(f"Doing {mesh}")

        # Load ground truth mesh
        filename = "data/" + mesh + ".obj"
        V_gt,F_gt = gpy.read_mesh(filename)
        V_gt = gpy.normalize_points(V_gt)
        # Create and abstract SDF function that is the only connection to the shape
        sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

        write_path = results_path + mesh +  "/"
        if not os.path.exists(write_path):
            os.makedirs(write_path)

        n = 120

        # Set up a grid
        gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
        U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
        S = sdf(U)

        # Marching cubes
        if method=="MC":
            print("  MC")
            V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
            gpy.write_mesh( write_path + "marching_cubes.obj", V_mc, F_mc)

        # Reach for the Spheres
        if method=="RFTS":
            print("  RFTS")
            if mesh in ["human_R", "human_A"]:
                # These segfault, so I am excluding them here.
                continue
            V0, F0 = gpy.icosphere(2)
            V_rfts, F_rfts = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)
            gpy.write_mesh( write_path + "rfts.obj", V_rfts, F_rfts)

        # Reach for the Arcs
        if method=="RFTA":
            print("  RFTA")
            V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True,
                max_points_per_sphere=3, fine_tune_iters=50,
                rng_seed = rng_seed)
            gpy.write_mesh( write_path + "rfta.obj", V_rfta, F_rfta)

        # Write gt
        gpy.write_mesh( write_path + "ground_truth.obj", V_gt, F_gt)
    