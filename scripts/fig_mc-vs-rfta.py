# This script replicates the results from Figure 4
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/mc-vs-rfta/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

def rotation_matrix(axis, angle):
    """
    Create a rotation matrix corresponding to the rotation around a general axis by a specified angle.

    R = dd^T + cos(theta)*(I - dd^T) + sin(theta)*skew(d)

    Parameters:
    axis : array
        Axis around which to rotate.
    angle : float
        Angle, in radians, by which to rotate.

    Returns:
    numpy.ndarray
        A rotation matrix.
    """

    # Ensure the axis is a unit vector
    axis = axis / np.linalg.norm(axis)

    # Components of the axis vector
    x, y, z = axis

    # Construct the skew-symmetric matrix
    skew_sym = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    # Identity matrix
    I = np.eye(3)

    # Outer product of the axis vector with itself
    outer = np.outer(axis, axis)

    # Rotation matrix
    R = outer + np.cos(angle) * (I - outer) + np.sin(angle) * skew_sym

    return R



mesh = "93366_sf"
# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)
# rotate V by a random amount along a random axis
np.random.seed(rng_seed)
axis = np.random.rand(3)
axis = axis / np.linalg.norm(axis)
angle = np.random.rand() * 2 * np.pi
# using numpy, rotation matrix from axis and angle
R = rotation_matrix(axis, angle)
V_gt = V_gt @ R
# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

ns = [ 10, 20, 50 ]
# if you have a better computer
# ns = [ 10, 20, 50, 100, 150 ]
    
for i, n in enumerate(ns):

    print(f"Running for n = {n}")

    # Set up a grid
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    # Marching squares
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

    # ndc
    # V_ndc, F_ndc = utility.ndc(V_gt, F_gt, n)

    # Reach for the Arcs
    V, F, P, N = rfta.reach_for_the_arcs(U, S, verbose = False, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=50, max_points_per_sphere=50)

    write_path = results_path + mesh +  f"/{n}/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    gpy.write_mesh( write_path + "ours.obj", V @ np.linalg.inv(R), F)
    gpy.write_mesh( write_path + "ground_truth.obj", V_gt @ np.linalg.inv(R), F_gt)
    gpy.write_mesh( write_path + "marching_cubes.obj", V_mc @ np.linalg.inv(R), F_mc)
    # gpy.write_mesh( write_path + "ndc.obj", V_ndc @ np.linalg.inv(R), F_ndc)
    
    # ps.init()
    # ps.register_surface_mesh("ours", V, F)
    # ps.register_surface_mesh("ground_truth", V_gt, F_gt)
    # ps.register_surface_mesh("marching_cubes", V_mc, F_mc)
    # ps.show()