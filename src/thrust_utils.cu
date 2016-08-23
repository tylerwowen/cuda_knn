/*
 * thrust_utils.cpp
 *
 *  Created on: Jul 21, 2016
 *      Author: tyler
 */

#include "thrust_utils.h"

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

static thrust::device_vector<short> *vector;

/**
 * Sort neighbors by distance
 */
void sortNeighbors(float *d_distances, int numNeighbors, short *d_indIdMap) {

  thrust::device_ptr<float> keys(d_distances);

  // initialize indices vector to [0,1,2,..]
  thrust::counting_iterator<int> iter(0);
  thrust::device_vector<short> ids(numNeighbors);
  thrust::copy(iter, iter + ids.size(), ids.begin());

  // first sort the keys and indices by the keys
  thrust::sort_by_key(keys, keys + numNeighbors, ids.begin(), thrust::greater<float>());
  thrust::copy(ids.begin(), ids.end(), thrust::device_pointer_cast(d_indIdMap));
}

void freeRawPointer(short *ptr) {
  vector->clear();
//  thrust::device_vector<short>().swap(*vector);
//  checkCudaErrors(cudaFree(dev_ptr));
}
