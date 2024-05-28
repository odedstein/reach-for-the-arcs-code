from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34323

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

results_path = "results/teaser_loewe/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

mesh = "loewe"
# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

np.random.seed(rng_seed)
axis = np.random.rand(3)
axis = axis / np.linalg.norm(axis)
angle = np.random.rand() * 2 * np.pi
# using numpy, rotation matrix from axis and angle
# Fixing the random angle chosen when creating this figure so we can run NDC
# on the remote computer and the same angle is guaranteed.
axis = np.array([0.23252653, 0.20370055, 0.95101919])
angle = 5.520514916060382
R = rotation_matrix(axis, angle)
V_gt = V_gt @ R

# Create an abstract SDF function that is the only connection to the shape
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
    # Silvia, please insert NDC code here.

    # Reach for the Spheres
    V0, F0 = gpy.icosphere(2)
    V_rfts, F_rfts = gpy.reach_for_the_spheres(U, None, V0, F0, S=S, verbose=False)

    # Reach for the Arcs
    V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True,
        fine_tune_iters=100, max_points_per_sphere=16)

    write_path = results_path + mesh +  f"/{n}/"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    gpy.write_mesh( write_path + "marching_cubes.obj", V_mc@R.T, F_mc)
    # gpy.write_mesh( write_path + "ndc.obj", V_ndc@R.T, F_ndc)
    gpy.write_mesh( write_path + "rfts.obj", V_rfts@R.T, F_rfts)
    gpy.write_mesh( write_path + "rfta.obj", V_rfta@R.T, F_rfta)
    gpy.write_mesh( write_path + "ground_truth.obj", V_gt@R.T, F_gt)
    