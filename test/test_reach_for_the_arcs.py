from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest
from .context import rfta
from .context import polyscope as ps
from .context import utility

class TestReachForTheArcs(unittest.TestCase):
    # this is not a great test, because it doesn't just check this function. But the general idea is "fine tuning should improve the result"
    def test_fine_tune_point_cloud(self):
        meshes = [ "boot.obj", "penguin.obj", "armadillo.obj", "horse.obj", "koala.obj" ]
        ns = [ 10, 20, 30 ]
        # rng
        rng_seed = 34523
        # meshes = [ "armadillo.obj" ]
        # ps.init()
        fine_tuning_was_worth_it = 0
        num_tests = 0
        for mesh in meshes:
            for n in ns:
                dist = np.Inf

                # Load ground truth mesh
                filename = "data/" + mesh
                V_gt,F_gt = gpy.read_mesh(filename)
                V_gt = gpy.normalize_points(V_gt)

                # Create and abstract SDF function that is the only connection to the shape
                sdf = lambda x: gpy.signed_distance(x, V_gt, F_gt)[0]

                # Set up a grid
                gx, gy, gz = np.meshgrid(np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1), np.linspace(-1.0, 1.0, n+1))
                U = np.vstack((gx.flatten(), gy.flatten(), gz.flatten())).T
                S = sdf(U)

                batch_sizes = [ 0 ]
                iters = [ 0, 5, 10, 20 ]
                for batch_size in batch_sizes[::-1]:
                    for iter in iters:
                        print(f"-- mesh = {mesh}, n = {n}, batch_size = {batch_size}, iters = {iter}")
                        V, F, P, N = rfta.reach_for_the_arcs(U, S,
                            fine_tune_iters = iter,
                            parallel = True, rng_seed = rng_seed,
                            return_point_cloud = True, verbose = False, batch_size = batch_size, local_search_t = 0.01)
                        # chamfer
                        chamfer_distance = utility.chamfer(V_gt, F_gt, V, F)
                        # assert chamfer_distance < dist
                        if chamfer_distance < dist:
                            fine_tuning_was_worth_it += 1
                        num_tests += 1
                        dist = chamfer_distance
                        print(f"chamfer_distance = {chamfer_distance}")
        # in the end, fine tuning should have improved the result *most of the time*
        print(f"fine_tuning_was_worth_it {fine_tuning_was_worth_it / num_tests} of the time")
        self.assertTrue(fine_tuning_was_worth_it / num_tests > 0.8)
            

            
            




        
        


if __name__ == '__main__':
    unittest.main()