# This script replicates the results from Figure 9
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/adding-spheres/"
# make directory if it doesnt exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/61258_sf.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 40
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Marching squares
V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

# Reach for the Arcs
V_with_spheres, F_with_spheres, P_with_spheres, N_with_spheres = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=30)

# without spheres
V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=1)

# without spheres, without finetuning
V_no_finetune, F_no_finetune, P_no_finetune, N_no_finetune = rfta.reach_for_the_arcs(U, S, verbose = True, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=0, max_points_per_sphere=1)

# Ideal point cloud and its reconstruction
_, I, b = gpy.squared_distance(U, V_gt, F_gt, use_cpp=True,use_aabb=True)
P_best = np.sum(V_gt[F_gt[I,:],:]*b[...,None], axis=1)
N_best = gpy.per_face_normals(V_gt, F_gt)[I,:]
V_best, F_best = rfta.point_cloud_to_mesh(P_best, N_best,
    screening_weight=10.0, verbose=False)

# write all meshes
gpy.write_mesh( results_path + "ground_truth.obj", V_gt, F_gt)
gpy.write_mesh( results_path + "marching_cubes.obj", V_mc, F_mc)
gpy.write_mesh( results_path + "ours.obj", V_with_spheres, F_with_spheres)
gpy.write_mesh( results_path + "ours_without_added_spheres.obj", V, F)
gpy.write_mesh( results_path + "ours_without_finetune.obj", V_no_finetune, F_no_finetune)
gpy.write_mesh( results_path + "best_point_cloud.obj", V_best, F_best)

# write all point clouds
gpy.write_mesh( results_path + "ours_point_cloud.ply", P_with_spheres, np.zeros((1,3)))
gpy.write_mesh( results_path + "ours_without_added_spheres_point_cloud.ply", P, np.zeros((1,3)))
gpy.write_mesh( results_path + "ours_without_finetune_point_cloud.ply", P_no_finetune, np.zeros((1,3)))
gpy.write_mesh( results_path + "best_point_cloud_point_cloud.ply", P_best, np.zeros((1,3)))

# find the added spheres with a tolerance of 1e-3
sphere_centers = U
sphere_radii = np.abs(S)
points_per_sphere = np.zeros(sphere_centers.shape[0], dtype=int)
points_one_per_sphere = []
added_points = []
for i in range(P_with_spheres.shape[0]):
    # find which sphere this point belongs to
    distances = np.linalg.norm(P_with_spheres[i,:] - sphere_centers, axis=1) - sphere_radii
    sphere_index = np.argmin(distances)
    # add this point to the sphere
    points_per_sphere[sphere_index] += 1
    if points_per_sphere[sphere_index] == 1:
        # this is the first point of this sphere
        points_one_per_sphere.append(P_with_spheres[i,:])
    if points_per_sphere[sphere_index] > 1:
        # this is the second point of this sphere
        added_points.append(P_with_spheres[i,:])

# now write a point cloud with one point per sphere
points_one_per_sphere = np.array(points_one_per_sphere)
gpy.write_mesh( results_path + "pre_added_spheres_point_cloud.ply", points_one_per_sphere, np.zeros((1,3)))
# and a point cloud of added spheres
added_points = np.array(added_points)
gpy.write_mesh( results_path + "added_spheres_point_cloud.ply", added_points, np.zeros((1,3)))