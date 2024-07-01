# This script generates the data for Figure 10
from context import *
import numpy as np
import gpytoolbox as gpy
import polyscope as ps

rng_seed = 34523

results_path = "results/runtime-empty-space/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Load ground truth mesh
filename = "data/fertility.obj"
V_gt, F_gt = gpy.read_mesh(filename)
V_gt = gpy.normalize_points(V_gt)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

# Set up a grid


types = ["rasterization-cpu", "rasterization-gpu", "rejection"]
ns = [ 10, 15, 20, 30, 45, 60, 80, 100 ]
times = np.zeros((len(types), len(ns)))
# for i in range(len(types)):
#     for j in range(len(ns)):
#         print("Running type: " + types[i] + " with n=" + str(ns[j]))
#         n = ns[j]
#         gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
#         U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
#         S = sdf(U)
#         times[i, j] = utility.aux_sample_empty_space(U, S, 1000, types[i])

# print(times)
# np.save(results_path + "times.npy", times)

# # plot the results
times = np.array([[6.38109922e-01, 1.84533787e+00, 3.98638630e+00, 1.24475420e+01,
  4.00137920e+01, 1.57115825e+02, 5.76655057e+02, 1.67031621e+03],
 [4.68774796e-01, 9.69583035e-01, 1.32345605e+00, 2.33696795e+00,
  4.00003886e+00, 8.48850608e+00, 1.92491331e+01, 3.70233929e+01],
 [3.38209867e-01, 2.72950101e+00, 3.51508570e+00, 1.72278910e+01,
  1.22903889e+02, 7.54325404e+02, 4.56859637e+03, 1.56528101e+04]])
# np.load(results_path + "times.npy")
print(times)
num_grid_cells = np.array(ns)**3
plt.figure()
plt.plot(num_grid_cells, times[0, :], label="rasterization-cpu")
plt.plot(num_grid_cells, times[1, :], label="rasterization-gpu")
plt.plot(num_grid_cells, times[2, :], label="rejection")
plt.xlabel("Number of points")
plt.ylabel("Time (s)")
# log scale
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.savefig(results_path + "runtime-empty-space.png")
# eps
plt.savefig(results_path + "runtime-empty-space.eps")
# pdf
plt.savefig(results_path + "runtime-empty-space.pdf")
plt.show()