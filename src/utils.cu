/*
 * utils.cu
 *
 *  Created on: Jul 22, 2016
 *      Author: tyler
 */

#include "utils.h"

/*
 * Kernel print function
 */
__global__ void printptr(short *ptr, int numNeighbors) {
  printf("ptr start\n");
  for (int x = 0; x < numNeighbors; x++) {
    printf("%d: %d; ", x, ptr[x]);
  }
  printf("\nptr end\n");
}

/*
 * Kernel Binary Search
 * @brief hello world
 *
 */
__device__ int isItemRatedAsc(int itemId, Rating *ratings, int numRatings) {
  int left = 0, right = numRatings - 1;

  while(left < right){
    int midInd = (left + right) / 2;
    unsigned midItemId = (ratings + midInd)->x;
    if (midItemId > itemId) {
      right = midInd - 1;
    }
    else if (midItemId < itemId) {
      left = midInd + 1;
    }
    else {
      return 1;
    }
  }
  // The last left==right matched
  if (itemId == (ratings + right)->x) return 1;
  return 0;
}

__device__ int isItemRatedDec(int itemId, Rating *ratings, int numRatings) {
  int left = 0, right = numRatings - 1;

  while(left < right){
    int midInd = (left + right) / 2;
    unsigned midItemId = (ratings + midInd)->x;
    if (midItemId > itemId) {
      left = midInd + 1;
    }
    else if (midItemId < itemId) {
      right = midInd - 1;
    }
    else {
      return 1;
    }
  }
  // The last left==right matched
  if (itemId == (ratings + right)->x) return 1;
  return 0;
}
