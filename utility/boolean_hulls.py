# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy
from gpytoolbox.copyleft import mesh_boolean
import sys, os
import polyscope as ps

# this will be a very bad wrapper of a bash script
def boolean_hulls(U,S):
    # we start with the mesh of a cube
    # this is a 8x3 matrix
    # V = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
    #               [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
    # V = V - 0.5
    # V = V * 2
    # F = np.array([[0,1,2],[0,2,3],[0,4,5],[0,5,1], # bottom
    #               [1,5,6],[1,6,2],[2,6,7],[2,7,3], # front
    #               [3,7,4],[3,4,0],[4,7,6],[4,6,5]]) # back
    # # change order, it's backwards
    # F = np.flip(F,axis=1)
    # # now, build a sphere of radius one
    # vs, fs = gpy.icosphere(2)
    # # now, loop over all U
    # for i in range(U.shape[0]):
    #     new_sphere = vs * np.abs(S[i])
    #     new_sphere = new_sphere + U[i,:]
    #     V, F = mesh_boolean(V,F,new_sphere,fs,boolean_type="difference")
    #     print("V.shape = ", F.shape)

    # ps.init()
    # ps.register_surface_mesh("hull", V, F)
    # # ps.register_surface_mesh("hull", vs, fs)
    # ps.show()
    # set up a fine grid
    n = 100
    gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
    U_big = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
    S_big_outer = np.zeros(U_big.shape[0]) + np.inf
    S_big_inner = np.zeros(U_big.shape[0]) + np.inf
    # now, loop over all U
    for i in range(U.shape[0]):
        # find grid cells that are contained in the sphere of radius S[i]
        distance_from_cell_to_sphere = np.linalg.norm(U_big - U[i,:],axis=1) - np.abs(S[i])
        if S[i] > 0:
            S_big_outer = np.minimum(S_big_outer, distance_from_cell_to_sphere)
        else:
            S_big_inner = np.minimum(S_big_inner, distance_from_cell_to_sphere)

    # now run marching cubes
    V_mc, F_mc = gpy.marching_cubes(S_big_outer, U_big, n+1, n+1, n+1)
    V_mc_in, F_mc_in = gpy.marching_cubes(S_big_inner, U_big, n+1, n+1, n+1)
    ps.init()
    ps.register_surface_mesh("outer", V_mc, F_mc)
    ps.register_surface_mesh("inner", V_mc_in, F_mc_in)
    ps.show() 