# This script generates the data for Figure 18
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/scorpion-and-horse/"
# make dir
if not os.path.exists(results_path):
    os.makedirs(results_path)

meshes = [ "horse", "scorpion" ]

for mesh in meshes:

    # Load ground truth mesh
    filename = "data/" + mesh + ".obj"
    # make dir
    if not os.path.exists(results_path + mesh):
        os.makedirs(results_path + mesh)
    V_gt,F_gt = gpy.read_mesh(filename)
    V_gt = gpy.normalize_points(V_gt)

    # Create and abstract SDF function that is the only connection to the shape
    sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

    # Set up a grid
    n = 50
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    # Marching squares
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

    # Reach for the Arcs
    V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=10)

    write_path = results_path + mesh + "/"

    # Write output
    gpy.write_mesh( write_path + "ours.obj", V, F)
    gpy.write_mesh( write_path + "ground_truth.obj", V_gt, F_gt)
    gpy.write_mesh( write_path + "marching_cubes.obj", V_mc, F_mc)


# Plot
# ps.init()
# ps.register_surface_mesh("ground truth", V_gt, F_gt)
# ps.register_surface_mesh("marching cubes", V_mc, F_mc)
# ps.register_surface_mesh("Reach for the Arcs", V, F)
# points = ps.register_point_cloud("Reach for the Arcs - points", P)
# points.add_vector_quantity("Reach for the Arcs - normals", N, enabled = True)
# ps.show()
