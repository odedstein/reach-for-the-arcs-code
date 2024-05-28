from context import *
import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt

# Option: smooth shape
filename = "data/illustrator.png"
poly = gpy.png2poly(filename)[0]
poly = poly[::5,:]

# Option: Batty's shape
# poly = 0.5*np.array([[-1.1,-1.1],[1.04,-1.04],[1.04,1.04],[-0.48,-0.48],[-1.06,1.06]])

# Set up polygon
poly = gpy.normalize_points(poly)
EC = gpy.edge_indices(poly.shape[0],closed=True)

# Need normals for PSR later
N = gpy.per_face_normals(poly, EC)

# Run various levels or point_cloud_to_mesh
ns = [20, 50, 100]
Vs = []
Fs = []
Ps = []
Ns = []
rng = np.random.default_rng(71926)
for n in ns:

    P,I,_ = gpy.random_points_on_mesh(poly, EC, n, rng=rng, return_indices=True)
    Ps.append(P)
    Ns.append(N[I,:])
    screening_weight = 100
    outer_boundary_type = "dirichlet"
    depth = 8
    V,F = rfta.point_cloud_to_mesh(P, Ns[-1], screening_weight=screening_weight,
        outer_boundary_type=outer_boundary_type,
        depth=depth,
        verbose=True)
    Vs.append(V)
    Fs.append(F)

def plot_edges(vv,ee,plt_str):
    ax = plt.gca()
    for i in range(ee.shape[0]):
        ax.plot([vv[ee[i,0],0],vv[ee[i,1],0]],
                 [vv[ee[i,0],1],vv[ee[i,1],1]],
                 plt_str,alpha=1)
def plot_pts(vv,nn,plt_str):
    s = 0.01
    ax = plt.gca()
    ax.scatter(vv[:,0], vv[:,1], c=plt_str)
    if nn is not None:
        ax.quiver(vv[:,0], vv[:,1], s*nn[:,0], s*nn[:,1], color=plt_str)

plt.figure()
plot_edges(poly, EC, 'k-')
cfgs = ['b', 'g', 'r', 'c']
for i in range(len(ns)):
    plot_edges(Vs[i], Fs[i], f'{cfgs[i]}-')
    # plot_pts(Ps[i], Ns[i], f'{cfgs[i]}')
plt.show()
