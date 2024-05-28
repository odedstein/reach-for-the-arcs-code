from context import *
import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt

from rfta_bindings import _sample_blue_points_on_spheres_cpp_impl

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
n = 16
gx, gy = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten())).T
S = sdf(U)

# Need normals for PSR later
N = gpy.per_face_normals(poly, EC)

# Run point_cloud_to_mesh with and without SDF info
n = 30
rng = np.random.default_rng(7926)
P,I,_ = gpy.random_points_on_mesh(poly, EC, n, rng=rng, return_indices=True)
# P = poly.copy()
# I = np.arange(P.shape[0])

N = N[I,:]
screening_weight = 10.
depth = 8
outer_boundary_type = "neumann"
V,F = rfta.point_cloud_to_mesh(P, N,
    screening_weight=screening_weight,
    outer_boundary_type=outer_boundary_type,
    depth=depth,
    verbose=True)

# Sample SDF to get inside and outside pts
known_inside_pts = _sample_blue_points_on_spheres_cpp_impl(
    U[S<0,:], np.abs(S[S<0]), 100.,
    rng.integers(0,np.iinfo(np.int32).max),
    False)
known_outside_pts = _sample_blue_points_on_spheres_cpp_impl(
    U[S>=0,:], S[S>=0], 100.,
    rng.integers(0,np.iinfo(np.int32).max),
    False)
known_weight = 1.
Vi,Fi = rfta.point_cloud_to_mesh(P, N,
    screening_weight=screening_weight,
    known_inside_pts=known_inside_pts,
    known_outside_pts=known_outside_pts,
    known_weight=known_weight,
    outer_boundary_type=outer_boundary_type,
    depth=depth,
    verbose=True)

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

# print(f"V: {V.shape}, Vi: {Vi.shape}")

plt.figure(1)
plot_edges(poly, EC, 'k-')
plot_edges(V, F, f'g-')
plot_spheres(U,S)
plot_pts(P, N, f'c')

plt.figure(2)
plot_edges(poly, EC, 'k-')
plot_pts(known_inside_pts, None, f'b')
plot_pts(known_outside_pts, None, f'r')
plot_edges(Vi, Fi, f'g-')
plot_spheres(U,S)
plot_pts(P, N, f'c')

plt.show()
