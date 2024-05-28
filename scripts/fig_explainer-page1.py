from context import *
import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt

# rng
rng_seed = 34523

results_path = "results/explainer-page1/"
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Option: smooth shape
filename = "data/cat.png"
# concatenate gpy.png2poly(filename)[0] and gpy.png2poly(filename)[1]
# poly = np.vstack((gpy.png2poly(filename)[0], gpy.png2poly(filename)[1]))
poly = gpy.png2poly(filename)[0]
poly = poly[::5,:]

# Option: Batty's shape
# poly = 0.5*np.array([[-1.1,-1.1],[1.04,-1.04],[1.04,1.04],[-0.48,-0.48],[-1.06,1.06]])

# Set up polygon
poly = 1.0*gpy.normalize_points(poly)
poly = poly + np.array([0.1, 0.0])[None,:]
EC = gpy.edge_indices(poly.shape[0],closed=True)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, poly, EC)[0]

# Set up a grid
n = 3
gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten())).T
S = sdf(U)

# Marching squares
V_ms, F_ms = gpy.marching_squares(S, U, n+1, n+1)

# Reach for the Arcs
V, F, P, N = rfta.reach_for_the_arcs(U, S,
    fine_tune_iters=20,
    parallel=False, rng_seed=rng_seed,
    return_point_cloud=True, verbose=True, max_points_per_sphere=10, screening_weight=10)

# save the outline, SDF data, point cloud, normals, and mesh as npy files
np.save(results_path + "outline.npy", poly)
np.save(results_path + "sdf_grid.npy", U)
np.save(results_path + "sdf.npy", S)
np.save(results_path + "point_cloud.npy", P)
np.save(results_path + "normals.npy", N)
np.save(results_path + "mesh_vertices.npy", V)
np.save(results_path + "mesh_faces.npy", F)

# plot everything
def plot_edges(vv,ee,plt_str):
    ax = plt.gca()
    for i in range(ee.shape[0]):
        ax.plot([vv[ee[i,0],0],vv[ee[i,1],0]],
                 [vv[ee[i,0],1],vv[ee[i,1],1]],
                 plt_str,alpha=1)
def plot_spheres(vv,f):
    ax = plt.gca()
    for i in range(vv.shape[0]):
        c = 'r' if f[i]>=0 else 'b'
        ax.add_patch(plt.Circle(vv[i,:], f[i], color=c, fill=False,alpha=0.1))
def plot_pts(vv,nn,plt_str):
    s = 0.01
    ax = plt.gca()
    ax.scatter(vv[:,0], vv[:,1], c=plt_str)
    if nn is not None:
        ax.quiver(vv[:,0], vv[:,1], s*nn[:,0], s*nn[:,1], color=plt_str)

plt.figure(figsize=(10,10))
plot_edges(poly,EC,'k')
plot_spheres(U,S)
plot_pts(P,N,'r')
plot_edges(V,F,'b')
plot_edges(V_ms,F_ms,'g')
plt.show()
