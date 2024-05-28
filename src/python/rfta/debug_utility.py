# Here I import only the functions I need for these functions
import numpy as np
import gpytoolbox as gpy

# checks that the point cloud and sdf data make sense
def assert_point_cloud_sdf_valid(points, normals, sdf_positions, sdf_vals, feasible_inds):
    # check that sdf_vals and sdf_positions are the same length
    assert len(sdf_vals) == len(sdf_positions)
    # check that points and normals are the same length
    assert points.shape[0] == normals.shape[0]
    # check that points and normals are same dimension
    assert points.shape[1] == normals.shape[1]
    # check that feasible_inds has same length as points
    assert feasible_inds.shape[0] == points.shape[0]
    # for each point, it should lay on the sphere with center sdf_vals[feasible_inds[i]] and radius sdf_positions[feasible_inds[i]]
    for i in range(feasible_inds.shape[0]):
        # print("i = %d" % i)
        # print("points[i,:] = %s" % points[i,:])
        # print("sdf_positions[feasible_inds[i],:] = %s" % sdf_positions[feasible_inds[i],:])
        # print("sdf_vals[feasible_inds[i]] = %f" % sdf_vals[feasible_inds[i]])
        # print("np.linalg.norm( points[i,:] - sdf_positions[feasible_inds[i],:] ) - np.abs(sdf_vals[feasible_inds[i]]) = %f" % (np.linalg.norm( points[i,:] - sdf_positions[feasible_inds[i],:] ) - np.abs(sdf_vals[feasible_inds[i]])))

        assert np.linalg.norm( points[i,:] - sdf_positions[feasible_inds[i],:] ) - np.abs(sdf_vals[feasible_inds[i]]) < 1e-3 # check that the point is on the sphere


def chamfer(v1,f1,v2,f2,n=100000):
    P1 = gpy.random_points_on_mesh(v1,f1,n)
    P2 = gpy.random_points_on_mesh(v2,f2,n)
    d1 = gpy.squared_distance(P1,P2,use_aabb=True,use_cpp=True)[0]
    d2 = gpy.squared_distance(P2,P1,use_aabb=True,use_cpp=True)[0]
    return np.sqrt(np.mean(d1)) + np.sqrt(np.mean(d2))