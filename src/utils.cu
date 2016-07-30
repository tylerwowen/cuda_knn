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
__device__ int findItemRating(int itemId, Rating *ratings, int numRatings) {
  if (numRatings < 1) return -1;
  int left = 0, right = numRatings - 1;
//  printf("itemid:%d, ratings[0].x:%d, ratings[0].y:%d, numratings:%d\n",itemId, ratings[0].x,
//      ratings[0].y, numRatings);
  while(left < right){
    int midInd = (left + right) / 2;
    unsigned midItemId = ratings[midInd].x;
    if (midItemId > itemId) {
      right = midInd - 1;
    }
    else if (midItemId < itemId) {
      left = midInd + 1;
    }
    else {
      return ratings[midInd].y;
    }
  }
  // The last left==right matched
  if (itemId == ratings[right].x) return ratings[right].y;
  return -1;
}


