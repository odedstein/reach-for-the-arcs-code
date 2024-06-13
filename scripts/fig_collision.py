# This script replicates the results from Figure 22
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

parser = argparse.ArgumentParser(description='Large quantitative experiment.')
parser.add_argument('--num_shapes', type=int, default=100, help='number of shapes to use')
parser.add_argument('--run', action=argparse.BooleanOptionalAction)
parser.set_defaults(run=False)
parser.add_argument('--metrics', action=argparse.BooleanOptionalAction)
parser.set_defaults(metrics=False)
args = parser.parse_args()

rng_seed = 34523

results_path = "results/collision/"
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


if args.run:

    mesh = "archer"
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

    n = 25

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

    write_path = results_path 
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    gpy.write_mesh( write_path + "ours.obj", V @ np.linalg.inv(R), F)
    gpy.write_mesh( write_path + "ground_truth.obj", V_gt @ np.linalg.inv(R), F_gt)
    gpy.write_mesh( write_path + "marching_cubes.obj", V_mc @ np.linalg.inv(R), F_mc)
    # gpy.write_mesh( write_path + "ndc.obj", V_ndc @ np.linalg.inv(R), F_ndc)

if args.metrics:
    num_points = 100000
    # generate num_points on the surface of a sphere
    # generate num_points by 3 array of normally distributed values with rng_seed
    np.random.seed(rng_seed)
    origins = np.random.normal(size=(num_points,3))
    # normalize the points
    origins = origins / np.linalg.norm(origins, axis=1)[:,None]
    # same thing for directions
    directions = np.random.normal(size=(num_points,3))
    directions = directions / np.linalg.norm(directions, axis=1)[:,None]
    # now, load the meshes
    V_gt, F_gt = gpy.read_mesh(results_path + "ground_truth.obj")
    # shoot rays from origins in directions towards the mesh
    ts_gt, _, _ = gpy.ray_mesh_intersect(origins, directions, V_gt, F_gt)
    # print(ts_gt)
    # turn ts_gt into binary "hit / no hit" (it is inf if no hit)
    ts_gt_bool = np.isfinite(ts_gt)

    meshes = ["marching_cubes", "ndc", "ours"]
    for mesh in meshes:
        # load
        filename = results_path + f"/{mesh}.obj"
        V,F = gpy.read_mesh(filename)
        # ts
        ts, _, _ = gpy.ray_mesh_intersect(origins, directions, V, F)
        # turn ts into binary "hit / no hit" (it is inf if no hit)
        ts_bool = np.isfinite(ts)
        # find number of false positives and false negatives by comparing to ts_gt_bool
        # proportion of false positives over all positives
        false_positives_prop = np.sum(np.logical_and(ts_bool, np.logical_not(ts_gt_bool))) / np.sum(ts_gt_bool)
        # proportion of false negatives over all negatives
        false_negatives_prop = np.sum(np.logical_and(np.logical_not(ts_bool), ts_gt_bool)) / np.sum(np.logical_not(ts_gt_bool))
        # print
        print(f"{mesh} false positives: {false_positives_prop}")
        print(f"{mesh} false negatives: {false_negatives_prop}")
        # proportion of all correct guesses (i.e., all that are not false positives or false negatives)
        correct_and_positive = np.logical_and(ts_bool, ts_gt_bool)
        correct_and_negative = np.logical_and(np.logical_not(ts_bool), np.logical_not(ts_gt_bool))
        correct_prop = np.sum(np.logical_or(correct_and_positive, correct_and_negative)) / num_points
        print(f"{mesh} correct: {correct_prop}")




