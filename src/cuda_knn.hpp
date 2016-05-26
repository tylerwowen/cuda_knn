/*
 * cuda_knn.hpp
 *
 *  Created on: May 24, 2016
 *      Author: tyler
 */

#ifndef CUDA_KNN_HPP_
#define CUDA_KNN_HPP_

#include "common.hpp"

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers);


#endif /* CUDA_KNN_HPP_ */
