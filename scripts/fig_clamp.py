from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/clamp"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/skull.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Set up a grid
n = 50
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T

# write ground truth mesh
gpy.write_mesh(os.path.join(results_path, "ground_truth.obj"), V_gt, F_gt)

clamp_values = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

for clamp_value in clamp_values:
    # log
    print(f"Clamp value: {clamp_value}")

    # Create and abstract SDF function that is the only connection to the shape
    sdf = lambda x: np.clip(gpy.signed_distance(x, V_gt, F_gt)[0], -clamp_value, clamp_value)
    S = sdf(U)

    # Marching squares
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
    # write mesh
    gpy.write_mesh(os.path.join(results_path, f"marching_cubes_{clamp_value}.obj"), V_mc, F_mc)

    # Reach for the Arcs
    V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = False, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=1, clamp_value=clamp_value)

    # write mesh
    gpy.write_mesh(os.path.join(results_path, f"clamp_{clamp_value}.obj"), V, F)