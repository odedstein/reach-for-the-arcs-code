from context import *

rng_seed = 34523

results_path = "results/noise/"
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




# Load ground truth mesh
filename = "data/54725_sf.obj"
# filename = "data/cheburashka.obj"
V_gt,F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

np.random.seed(rng_seed)
axis = np.random.rand(3)
axis = axis / np.linalg.norm(axis)
angle = np.random.rand() * 2 * np.pi
# using numpy, rotation matrix from axis and angle
R = rotation_matrix(axis, angle)
V_gt = V_gt @ R

gpy.write_mesh(results_path + "ground-truth.obj", V_gt @ np.linalg.inv(R), F_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid
n = 25
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
S = sdf(U)

# Marching squares
V_mc, F_mc = gpy.marching_cubes(S, U, n+1, n+1, n+1)
gpy.write_mesh(results_path + "marching-cubes.obj", V_mc @ np.linalg.inv(R), F_mc)
# chamfer error
chf_mc = rfta.chamfer(V_gt, F_gt, V_mc, F_mc)
print("chamfer error (marching cubes) = {}".format(chf_mc))
# hausdorff distance
hd_mc = gpy.approximate_hausdorff_distance(V_gt, F_gt, V_mc, F_mc)

# Reach for the Arcs
# screening weights powers of 2 from 2^{-4} to 2^{8}
# noise_amplitudes = [0.0, 0.001, 0.01, 0.02, 0.05, 0.1]
noise_amplitudes = []
for i, amp in enumerate(noise_amplitudes):
    # log
    print("noise amplitude = {}".format(amp))
    S_with_noise = S + amp * np.random.randn(S.shape[0])
    V, F, P, N = rfta.reach_for_the_arcs(U, S_with_noise, verbose = False, parallel = True, return_point_cloud=True, debug_Vgt=V_gt, debug_Fgt=F_gt, fine_tune_iters=20, max_points_per_sphere=30)
    # write output
    gpy.write_mesh(results_path + "ours-{}.obj".format(amp), V @ np.linalg.inv(R), F)
    # print chamfer error
    chf = rfta.chamfer(V_gt, F_gt, V, F)
    print("chamfer error (ours), noise amp {} = {}".format(amp,chf))

    # also marching cubes
    V_mc, F_mc = gpy.marching_cubes(S_with_noise, U, n+1, n+1, n+1)
    gpy.write_mesh(results_path + "marching-cubes-{}.obj".format(amp), V_mc @ np.linalg.inv(R), F_mc)

    # also rfts
    # V0, F0 = gpy.icosphere(2)
    # V, F = gpy.reach_for_the_spheres(U, None, V0, F0, S=S_with_noise, verbose=False)
    # gpy.write_mesh(results_path + "rfts-{}.obj".format(amp), V @ np.linalg.inv(R), F)

    # V_ndc, F_ndc = utility.ndc(V_gt, F_gt, n)

# bar chart of chamfer errors
        
chmf_errors = [0.075, 0.035, 0.038, 0.04, 0.057, 0.15]
plt.bar(range(len(chmf_errors)), chmf_errors)
plt.xlim([-1, len(chmf_errors)])
plt.ylim([0, 0.2])
plt.savefig(results_path + "chamfer_errors.eps")
plt.close()