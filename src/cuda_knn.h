/*
 * cuda_knn.h
 *
 *  Created on: May 24, 2016
 *      Author: tyler
 */

#ifndef CUDA_KNN_H_
#define CUDA_KNN_H_

#include "common.h"

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers);


#endif /* CUDA_KNN_H_ */
