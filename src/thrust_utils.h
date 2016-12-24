/*
 * thrust_utils.h
 *
 *  Created on: Jul 21, 2016
 *      Author: tyler
 */

#ifndef THRUST_UTILS_H_
#define THRUST_UTILS_H_


void sortNeighbors(float *d_distances, int numNeighbors, int *d_indIdMap);
void freeRawPointer(short *ptr);

#endif /* THRUST_UTILS_H_ */
