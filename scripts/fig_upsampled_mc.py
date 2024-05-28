# Thanks to our context.py file, we only need this in the header:
from context import * 

# Set up gt
# v, f = gpy.read_mesh('data/alice_rb.stl') 
# v, f = gpy.read_mesh('data/table/lucy.obj')
# v, f = gpy.read_mesh('data/scorpion.obj')
v, f = gpy.read_mesh('data/cat-low-resolution.obj')
# v, f = gpy.read_mesh('data/fandisk-fine.obj')
# v, f, _, _ = gpy.decimate(v, f, face_ratio=0.1)
# v, f = gpy.read_mesh('data/cube.obj')
v = gpy.normalize_points(v)

# Create and abstract SDF function that is the only connection to the shape
sdf = lambda x: gpy.signed_distance(x, v, f)[0]


# Set up a grid and do marching squares for initial guess
n = 20
gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
h = 2.0/n
GV = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
# V0, E0 = gpy.marching_squares(sdf(GV), GV, n+1, n+1)
sdf_vals = sdf(GV)
V_mc, F_mc = gpy.marching_cubes(sdf_vals, GV, n+1, n+1, n+1)

ns_upsampled = [20,40,80,160,320]
for ns in ns_upsampled:
    print('upsampling to {}'.format(ns))
    fd_gx, fd_gy, fd_gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), indexing='ij')
    fd_GV = np.concatenate([fd_gx.reshape(-1,1,order='F'), fd_gy.reshape(-1,1,order='F'), fd_gz.reshape(-1,1,order='F')], axis=1)
    fd_sdf_vals = sdf(fd_GV)
    gx_2, gy_2, gz_2 = np.meshgrid(np.linspace(-1.0, 1.0, ns+1), np.linspace(-1.0, 1.0, ns+1), np.linspace(-1.0, 1.0, ns+1))
    GV_2 = np.vstack((gx_2.flatten(), gy_2.flatten(), gz_2.flatten())).T
    W = gpy.fd_interpolate(GV_2,np.array([n+1,n+1,n+1],dtype=np.int32),h,corner=np.array([-1,-1,-1]))
    sdf_vals_upsampled = W @ fd_sdf_vals
    V_mc_upsampled, F_mc_upsampled = gpy.marching_cubes(sdf_vals_upsampled, GV_2, ns+1, ns+1, ns+1)
    gpy.write_mesh('results/upsample-mc/mc-{}.obj'.format(ns), V_mc_upsampled, F_mc_upsampled)

# ps.init()
# ps.register_surface_mesh("ground truth", v, f)
# ps.register_point_cloud("grid-inside", GV[sdf(GV) < 0.0])
# ps.show()


# V0, F0 = gpy.marching_cubes(sdf(GV)+0.00, GV, n+1, n+1, n+1)

# V0, F0 = V_mc, F_mc
# flip triangles
# F0[:,[0,1,2]] = F0[:,[0,2,1]]
# V0, F0 = gpy.remesh_botsch(V0, F0, h=0.2, i=10)

V0, F0 = gpy.icosphere(2)
# V0 = 0.8*V0
# V0 = 0.5*V0
# V0 = 2.0*V0
# F0 = f.copy()
# min_h = None #np.minimum(2.0/n,0.1)
# min_h = 0.01
save_dir = 'results/upsample-mc/'
# mesh_exporter_callback = utility.mesh_exporter(save_dir, 1)
# def callback(state):
#     mesh_exporter_callback(state)
# U,G = src.sdf_flow(GV, sdf, V0, F0, max_iter=2000, h=0.2, tol=1e-3, resample=0, 
#                    feature_detection='aggressive',
#     inside_outside_test=True, output_sensitive=True, visualize=True, 
#     remesh_iterations=1,min_h=0.05)

V_rfta, F_rfta = rfta.reach_for_the_arcs(GV, sdf_vals, verbose=False, parallel=True, fine_tune_iters=2, max_points_per_sphere=3)

gpy.write_mesh(save_dir+'/rfta.obj', V_rfta, F_rfta)

V_rfts, F_rfts = gpy.read_mesh(save_dir+'/rfts.obj')

# gpy.write_mesh(save_dir+'/initial.obj', V0, F0)
# gpy.write_mesh(save_dir+'/ours.obj', U, G)
# gpy.write_mesh(save_dir+'/ground_truth.obj', v, f)
# gpy.write_mesh(save_dir+'/marching_cubes.obj', V_mc, F_mc)

# U,G = src.sdf_flow(GV, sdf, U, G, v,f, max_iter=2000, h=0.1, tol=1e-3, resample=0, feature_detection='aggressive',
#     inside_outside_test=True, output_sensitive=True, visualize=True, dt=20.0, remesh_iterations=1, min_h = 0.05)
# # gpy.write_mesh('debug.obj'    , U, G)
# # U, G = gpy.read_mesh('debug.obj')
# U,G = src.sdf_flow(GV, sdf, U, G, v,f, max_iter=2000, h=0.05, tol=1e-3, resample=0, feature_detection='aggressive',
#     inside_outside_test=True, output_sensitive=True, visualize=True, dt=40.0, remesh_iterations=1, min_h = 0.02)
# U,G = src.sdf_flow(GV, sdf, U, G, v,f, max_iter=2000, h=0.02, tol=1e-3, resample=0, feature_detection='aggressive',
#     inside_outside_test=True, output_sensitive=True, visualize=True, dt=80.0, remesh_iterations=1, min_h = 0.01)
ps.init()
ps.register_surface_mesh("ground truth", v, f)
ps.register_surface_mesh("marching cubes", V_mc, F_mc)
ps.register_surface_mesh("initial guess", V0, F0)
ps.register_surface_mesh("ours", V_rfta, F_rfta)
ps.register_surface_mesh("rfts", V_rfts, F_rfts)

# # Do gridless SDF
# # n = 20
# # gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
# # GV_big = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
# # GV = GV_big.copy()
# GV = np.vstack((GV,np.random.rand(1000,3) * 2.0 - 1.0))
# U,G = src.sdf_to_mesh(GV, sdf(GV), U, G, max_iter=20000, poly=v, EC=f, h=0.02)
# ps.register_surface_mesh("MC + more samples", U, G)

ps.show()