from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest
from .context import rfta

class TestLocallyMakeFeasible(unittest.TestCase):
    def test_random_spheres(self):
        """Test the random spheres function in 2D"""
        for d in [2,3]:
            for i in range(20):
                # for par in [True,]
                # print((i, d))
                rng_seed = i
                rng = np.random.RandomState(rng_seed)
                # random value between 10 and 10000
                n_spheres = rng.randint(10,1000)
                # n_spheres = 100
                tol = 1e-4
                
                res = rng.randint(20,100)
                centers = rng.uniform(size=(n_spheres,d))
                radii = 0.05*rng.uniform(size=(n_spheres,))
                # generate a random point on each sphere
                random_points_on_spheres = rng.uniform(size=(n_spheres,d))
                random_points_on_spheres = random_points_on_spheres/np.linalg.norm(random_points_on_spheres,axis=1)[:,None]
                random_points_on_spheres = centers + radii[:,None]*random_points_on_spheres
                #
                P, N, f = rfta.locally_make_feasible(centers, radii, random_points_on_spheres, rng_seed = rng_seed, batch_size = 0, tol = tol, parallel = False, verbose = False)
                # how many points we generated
                self.assertTrue(P.shape[1]==d)
                self.assertTrue(P.shape[0]>0)
                # print("Generated %d points out of %d spheres." % (P.shape[0],centers.shape[0]))
                # is every generated point on the f[i]-th sphere?
                for i in range(P.shape[0]):
                    # assert that the point is *on* the sphere, not inside
                    self.assertTrue(np.abs(np.linalg.norm(P[i,:]-centers[f[i],:]) - radii[f[i]]) < tol)
                    # also: should be outside all other spheres
                    for j in range(n_spheres):
                        if j!=f[i]:
                            self.assertTrue(np.linalg.norm(P[i,:]-centers[j,:]) > radii[j] - tol)
                # check that if we use parallelization we get the same result
                P_par, N_par, f_par = rfta.locally_make_feasible(centers, radii, random_points_on_spheres, rng_seed = rng_seed, batch_size = 0, tol = tol, parallel = True, verbose = False)
                self.assertTrue(np.all(f==f_par))
                self.assertTrue(np.all(np.abs(P-P_par)<tol))
                self.assertTrue(np.all(np.abs(N-N_par)<tol))




        
        


if __name__ == '__main__':
    unittest.main()