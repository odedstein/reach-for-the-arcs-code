#include <Eigen/Core>
#include <random>
#include "outside_points_from_rejection_sampling.h"
#include "sAABB.h"
#include <iostream>

template<int dim>
void outside_points_from_rejection_sampling(
    const Eigen::MatrixXd & sdf_points,
    const Eigen::MatrixXd & sphere_radii,
    const int rng_seed,
    const int num_samples,
    Eigen::MatrixXd & samples){
        // std::cout << "rejection_sample_outside_spheres" << std::endl;
    using Vec = Eigen::Matrix<double, dim, 1>;
    std::vector<Vec> samples_vector;
    Eigen::VectorXi batch;
    batch.resize(0);
    const sAABB<dim> aabb_tree(sdf_points,
        sphere_radii.array().abs(), 1e-6);
    auto volumetric_sampling_start = std::chrono::high_resolution_clock::now();
    int attempts_counter = 0;
    // std::default_random_engine generator;
//    std::default_random_engine generator(seed);
//    std::srand(seed);
    std::mt19937 rng(rng_seed);
    
    while(samples_vector.size() < num_samples){
        attempts_counter++;
        // sample a random point in [-1,1]^3
        Vec point;
//        point.setRandom();
        std::uniform_real_distribution<double> r_unif(-1.,1.);
        for(int i=0; i<point.size(); ++i) {
            point[i] = r_unif(rng);
        }
        // point = point.array()*2.0;
        // point = point.array() - 1.0;
        // std::cout << "point: " << point << std::endl;
        // get the minimum distance to the spheres
        std::vector<int> prims;
        int min_primitive;
        aabb_tree.get_spheres_containing(point, 1, -1e-6, -1,
                prims);
        // this -> get_min_distance(sphere_centers, sphere_radii, point, -10.0, min_distance, min_primitive);
        // std::cout << "min_distance: " << min_distance << std::endl;
        // std::cout << "min_primitive: " << min_primitive << std::endl;
        // std::cout << "Radius of min_primitive: " << sphere_radii(min_primitive) << std::endl;
        // std::cout << "Center of min_primitive: " << sphere_centers.row(min_primitive) << std::endl;
        // if the minimum distance is positive, then the point is outside the spheres
        if(prims.empty()){
            samples_vector.push_back(point);
        }
    }
    // std::cout << "samples_vector.size(): " << samples_vector.size() << std::endl;
    samples.resize(samples_vector.size(), 3);
    for(int i = 0; i < samples_vector.size(); i++){
        samples.row(i) = samples_vector[i];
    }
    auto volumetric_sampling_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = volumetric_sampling_end - volumetric_sampling_start;
    // std::cout << "Using rejection sampling, I generated " << samples_vector.size() << " samples in " << attempts_counter << " attempts and " << elapsed.count() << " seconds." << std::endl;
    }
    
    template void outside_points_from_rejection_sampling<2>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, int, int, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);
    template void outside_points_from_rejection_sampling<3>(Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, Eigen::Matrix<double, -1, -1, 0, -1, -1> const&, int, int, Eigen::Matrix<double, -1, -1, 0, -1, -1>&);