#ifndef OUTSIDE_POINTS_FROM_REJECTION_SAMPLING_H
#define OUTSIDE_POINTS_FROM_REJECTION_SAMPLING_H

#include <Eigen/Core>

template<int dim>
void outside_points_from_rejection_sampling(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    const int num_samples,
    Eigen::MatrixXd & samples);

#endif