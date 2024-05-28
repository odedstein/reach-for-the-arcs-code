from context import *
import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt

# rng
rng_seed = 34523

# Option: smooth shape
# filename = "data/illustrator.png"
# poly = gpy.png2poly(filename)[0]
# poly = poly[::5,:]

# Option: Batty's shape
poly = 0.5*np.array([[-1.1,-1.1],[1.04,-1.04],[1.04,1.04],[-0.48,-0.48],[-1.06,1.06]])

# Set up polygon
poly = gpy.normalize_points(poly)
EC = gpy.edge_indices(poly.shape[0],closed=True)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, poly, EC)[0]

# Set up a grid
n = 12
gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten())).T
S = sdf(U)

# Marching squares
V_ms, F_ms = gpy.marching_squares(S, U, n+1, n+1)

# Reach for the Arcs
V, F, P, N = rfta.reach_for_the_arcs(U, S,
    fine_tune_iters=5,
    parallel=False, rng_seed=rng_seed,
    return_point_cloud=True, verbose=True)

# Ideal point cloud and its reconstruction
_, I, b = gpy.squared_distance(U, poly, EC, use_cpp=True,use_aabb=True)
P_best = np.sum(poly[EC[I,:],:]*b[...,None], axis=1)
N_best = gpy.per_face_normals(poly, EC)[I,:]
V_best, F_best = rfta.point_cloud_to_mesh(P_best, N_best,
    screening_weight=10., verbose=False)

# Plot
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

plt.figure(1)
plot_spheres(U,S)
plot_edges(poly, EC, 'k-')
plot_edges(V_ms, F_ms, 'm-')
plt.title('Marching squares')

plt.figure(2)
plot_spheres(U,S)
plot_edges(poly, EC, 'k-')
plot_edges(V, F, 'g-')
plot_pts(P, N, f'g')
plt.title('Reach for the Arcs')

plt.figure(3)
plot_spheres(U,S)
plot_edges(poly, EC, 'k-')
plot_edges(V_best, F_best, 'b-')
plot_pts(P_best, N_best, f'b')
plt.title('Best point cloud')

plt.show()
