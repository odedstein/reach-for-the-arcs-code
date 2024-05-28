from context import *
import numpy as np
import gpytoolbox as gpy
import matplotlib.pyplot as plt

seed = 4623452

# Option: smooth shape
filename = "data/illustrator.png"
poly = gpy.png2poly(filename)[0]
poly = poly[::5,:]

# Option: Batty's shape
# poly = 0.5*np.array([[-1.1,-1.1],[1.04,-1.04],[1.04,1.04],[-0.48,-0.48],[-1.06,1.06]])

# Set up polygon
poly = 0.5*gpy.normalize_points(poly) + np.array([[0.5, 0.5]])
EC = gpy.edge_indices(poly.shape[0],closed=True)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, poly, EC)[0]

# Set up a grid
n = 12
gx, gy = np.meshgrid(np.linspace(0., 1.0, n+1), np.linspace(0., 1.0, n+1))
U = np.vstack((gx.flatten(), gy.flatten())).T
S = sdf(U)

# Rasterize
neg = S<0
pos = np.logical_not(neg)
res = 360 #16
print("Negative spheres...")
P_neg = rfta.outside_points_from_rasterization(U[neg,:], S[neg], rng_seed=seed,
    narrow_band=True,
    res=res, verbose=True)
print("Positive spheres...")
P_pos = rfta.outside_points_from_rasterization(U[pos,:], S[pos], rng_seed=seed,
    narrow_band=True,
    res=res, verbose=True)

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

fig1 = plt.figure(1)
plot_spheres(U,S)
plot_edges(poly, EC, 'k-')
plot_pts(P_neg, None, 'g')

fig2 = plt.figure(2)
plot_spheres(U,S)
plot_edges(poly, EC, 'k-')
plot_pts(P_pos, None, 'g')

plt.show()
