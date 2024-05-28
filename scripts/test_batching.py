from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

# rng
rng_seed = 34523

# Load ground truth mesh
filename = "data/armadillo.obj"
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
# V, F, P, N = rfta.reach_for_the_arcs(U, S,
#     fine_tune_iters = 10,
#     parallel = True, rng_seed = rng_seed,
#     return_point_cloud = True, verbose = True, batch_size = 10000)
# chamfer_distance = utility.chamfer(V_gt, F_gt, V, F)
# print(f"chamfer_distance = {chamfer_distance}")
# batch_sizes = [ 100, 1000, 10000, 0 ]
# iters = [ 0, 2, 4 ]
batch_sizes = [ 10000 ]
iters = [ 0, 1, 2, 10, 50 ]
# iters = [ 50 ]
for batch_size in batch_sizes[::-1]:
    for iter in iters:
        print(f"-- batch_size = {batch_size}, iters = {iter}")
        V, F, P, N = rfta.reach_for_the_arcs(U, S,
            fine_tune_iters = iter,
            parallel = True, rng_seed = rng_seed,
            return_point_cloud = True, verbose = True, batch_size = batch_size, local_search_t = 0.01, num_rasterization_spheres = batch_size, debug_Vgt = V_gt, debug_Fgt = F_gt, screening_weight = 0.1)
        # chamfer
        chamfer_distance = utility.chamfer(V_gt, F_gt, V, F)
        print(f"chamfer_error = {chamfer_distance}")

# Plot
ps.init()
ps.register_surface_mesh("ground truth", V_gt, F_gt, enabled = False)
ps.register_surface_mesh("marching cubes", V_mc, F_mc, enabled = False)
ps.register_surface_mesh("Reach for the Arcs", V, F)
points = ps.register_point_cloud("Reach for the Arcs - points", P)
points.add_vector_quantity("Reach for the Arcs - normals", N, enabled = True)
ps.show()
