from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 2230

results_path = "results/sharp_features/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

mesh = "pyramid"
# Load ground truth mesh
filename = "data/" + mesh + ".obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

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

rng = np.random.default_rng(rng_seed)
axis = rng.random(3)
axis = axis / np.linalg.norm(axis)
angle = rng.random() * 2 * np.pi
# using numpy, rotation matrix from axis and angle
R = rotation_matrix(axis, angle)
V_gt = V_gt @ R

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

ns = [40]
for n in ns:
    print(f"doing n={n}")
    # Set up gt

    # Set up a grid
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S = sdf(U)

    # Marching cubes
    print("  MC")
    V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)

    # Reach for the Arcs
    print("  RFTA")
    V_rfta, F_rfta = rfta.reach_for_the_arcs(U, S, verbose=False, parallel=True,
        rng_seed=rng_seed)


    write_path = f"{results_path}/{n}"
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    # Write output
    gpy.write_mesh( write_path + f"/rfta.obj", V_rfta@R.T, F_rfta)
    gpy.write_mesh( write_path + f"/ground_truth.obj", V_gt@R.T, F_gt)
    gpy.write_mesh( write_path + f"/mc.obj", V_mc@R.T, F_mc)


    