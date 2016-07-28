/*
 * utils.cuh
 *
 *  Created on: Jul 22, 2016
 *      Author: tyler
 */

#ifndef UTILS_CUH_
#define UTILS_CUH_

#include "common.cuh"

__global__ void printptr(short *ptr, int numNeighbors);
__device__ int isItemRated(int itemId, Rating *ratings, int numRatings);

#endif /* UTILS_CUH_ */
