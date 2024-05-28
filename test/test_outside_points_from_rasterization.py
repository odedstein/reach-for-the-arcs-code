from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import scipy as sp
from .context import unittest
from .context import rfta

class TestOutsidePointsFromRasterization(unittest.TestCase):
    def test_random_spheres(self):
        """Test the random spheres function in 2D"""
        for d in [2,3]:
            for i in range(20):
                # print((i, d))
                rng_seed = i
                rng = np.random.RandomState(rng_seed)
                # random value between 10 and 10000
                n_spheres = rng.randint(10,1000)
                # n_spheres = 100
                tol = 1e-4
                
                res = rng.randint(20,100)
                centers = rng.uniform(size=(n_spheres,d))
                radii = 0.07*rng.uniform(size=(n_spheres,))
                P = rfta.outside_points_from_rasterization(centers, radii, rng_seed=rng_seed, tol=tol, res=res, parallel=False)
                # print("Generated %d points." % P.shape[0])
                self.assertTrue(P.shape[1]==d)
                self.assertTrue(P.shape[0]>0) # making sure we're not checking something dumb because no point is generated.
                # for every row in P, assert it's not in any sphere
                error_margin = tol + (2 * np.sqrt(d) / res)
                for i in range(P.shape[0]):
                    self.assertTrue(np.all(np.linalg.norm(P[i,:]-centers, axis=1)>(radii - error_margin)))
                # also, the rasterized points densely cover space; in other words, every point in [0,1]^d is either inside a sphere or  within 2 * sqrt(d) / res of a sample
                for i in range(100):
                    p = rng.uniform(size=(1,d))
                    is_in_a_sphere = np.any(np.linalg.norm(p-centers, axis=1)<(radii + error_margin))
                    if not is_in_a_sphere:
                        self.assertTrue(np.min(np.linalg.norm(P-p, axis=1)) <  2 * np.sqrt(d) / res)
                # if parallel, same
                P_par = rfta.outside_points_from_rasterization(centers, radii, rng_seed=rng_seed, tol=tol, res=res, parallel = True)
                self.assertTrue(np.all(np.abs(P-P_par)<tol))

        
        


if __name__ == '__main__':
    unittest.main()