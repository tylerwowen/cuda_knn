/*
 * utils.h
 *
 *  Created on: Jul 22, 2016
 *      Author: tyler
 */

#ifndef UTILS_H_
#define UTILS_H_

#include "common.cuh"

__global__ void printptr(short *ptr, int numNeighbors);
__device__ int isItemRatedAsc(int itemId, Rating *ratings, int numRatings);
__device__ int isItemRatedDec(int itemId, Rating *ratings, int numRatings);

#endif /* UTILS_H_ */
