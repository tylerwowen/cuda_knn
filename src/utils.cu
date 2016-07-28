/*
 * utils.cu
 *
 *  Created on: Jul 22, 2016
 *      Author: tyler
 */

#include "utils.cuh"

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
 * Kernel Binary Search an ascending array
 * @return  -1 if not found, otherwise returns the corresponding rating
 *
 */
__device__ int isItemRated(int itemId, Rating *ratings, int numRatings) {
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

