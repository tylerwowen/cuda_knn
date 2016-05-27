/*
 * common.cuh
 *
 *  Created on: May 19, 2016
 *      Author: tyler
 */

#ifndef COMMON_CUH_
#define COMMON_CUH_

#include "helper_cuda.h"

/**
 *  {itemId, rating}
 */
typedef struct __align__(8) {
  unsigned itemId;
  float rating;
} Rating;

/**
 *  {RatingsPtr, numRatings}
 */
typedef struct __align__(8) {
  Rating *ratings;
  int numRatings;
} User;

#endif /* COMMON_CUH_ */
