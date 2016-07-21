/*
 ============================================================================
 Name        : cuda_knn.cu
 Author      : Tyler Ouyang
 Version     :
 Copyright   : Copyright © 2016 Tyler Ouyang. All rights reserved.
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include <iostream>
#include <stdlib.h>

#include "common.cuh"
#include "common.h"
#include "thrust_utils.h"

using namespace std;

#define TILE_SIZE 24
#define TILE_DEPTH 256

__device__
float calculateCOSDistance(
    Rating *r1Start,
    Rating *r1End,
    Rating *r2Start,
    Rating *r2End) {

  float dotProduct = 0.0, r1NormSQ = 0.0, r2NormSQ = 0.0;

  while (r1Start < r1End && r2Start < r2End) {
    if (r1Start->x > r2Start->x) {
      // treat r1Start->rating as 0
      r2NormSQ += r2Start->y * r2Start->y;
      r2Start++;
    } else if (r1Start->x == r2Start->x) {
      dotProduct += r1Start->y * r2Start->y;
      r1NormSQ += r1Start->y * r1Start->y;
      r2NormSQ += r2Start->y * r2Start->y;
      r1Start++;
      r2Start++;
    } else {
      // treat r2Start->y as 0
      r1NormSQ += r1Start->y * r1Start->y;
      r1Start++;
    }
  }
  // finish baseUser tail, if any
  while (r1Start < r1End) {
    r1NormSQ += r1Start->y * r1Start->y;
    r1Start++;
  }
  // finish neighbor tail, if any
  while (r2Start < r2End) {
    r2NormSQ += r2Start->y * r2Start->y;
    r2Start++;
  }
  // distance
  return dotProduct / (sqrt(r1NormSQ) * sqrt(r2NormSQ));
}

/**
 * CUDA kernel that computes distances between every two users in d_users
 */
__global__
void calculateAllDistance(
    int stageStartUser,
    int numUsers,
    int *d_users,
    Rating *allRatings,
    float *d_distances) {

  int globalUserId = stageStartUser + blockIdx.x * blockDim.x + threadIdx.x;
  // user id in stage
  int localUserId = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: experimental, need optimization
  // space for TILE_SIZE * 2 users, each one has at most TILE_DEPTH ratings
  __shared__ Rating ratings[TILE_DEPTH * TILE_SIZE * 2];

  int baseNumRatings = d_users[globalUserId];
  int numRatings = min(baseNumRatings, TILE_DEPTH);
  Rating *baseStart = ratings + (threadIdx.x + TILE_SIZE) * TILE_DEPTH, *baseEnd = baseStart + numRatings;

  // copy data to shared memory, base users are the last TILE_SIZE users in ratings
  Rating *copyFrom = allRatings + globalUserId * TILE_DEPTH;
#pragma unroll
  for (int i = threadIdx.y; i < numRatings; i += TILE_SIZE)
    baseStart[i] = copyFrom[i];
  __syncthreads();

//  printf("hello from block %d thread x %d, thread y %d\n", blockIdx.x, threadIdx.x, threadIdx.y);

  int *tileStartUser = d_users;
  // TILE_SIZE user per time for now
  for (int i = 0; i < numUsers; i += TILE_SIZE, tileStartUser += TILE_SIZE) {
    int neighborNumRatings = tileStartUser[threadIdx.y];
    int nbNumRatings = min(neighborNumRatings, TILE_DEPTH);
    Rating *neighborStart = ratings + threadIdx.y * TILE_DEPTH, *neighborEnd = neighborStart + nbNumRatings;

    copyFrom = allRatings + (i + threadIdx.y) * TILE_DEPTH;
    // copy data to shared memory, neighbors are the first TILE_SIZE users
#pragma unroll
    for (int j = threadIdx.x; j < nbNumRatings; j += TILE_SIZE)
      neighborStart[j] = copyFrom[j];
    // TODO: what if there are more than TILE_DEPTH users
    __syncthreads();

    d_distances[localUserId * numUsers + i + threadIdx.y] = calculateCOSDistance(baseStart, baseEnd, neighborStart,
        neighborEnd);
    __syncthreads();
  }

  printf("distance from user %d to user %d is %.20lf\n", localUserId, threadIdx.y,
      d_distances[localUserId * numUsers + threadIdx.y]);
}

/**
 * CUDA kernel that computes KNN
 */
__global__
void knn() {

}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    int **d_users,
    Rating **d_ratings) {

  // make numTrainUsers a multiple of TILE_SIZE
  int numTrainUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  int numRatings = numTrainUsers * TILE_DEPTH;
  int *h_users = new int[numTrainUsers];
  ;
  for (int i = 0; i < numTrainUsers; i++)
    h_users[i] = 0;

  Rating *h_ratings = new Rating[sizeof(Rating) * numRatings];
  checkCudaErrors(cudaMalloc((void **) d_ratings, sizeof(Rating) * numRatings));
  cout << "total size of ratings in bytes: " << sizeof(Rating) * numRatings << endl;

  for (int i = 0; i < numTrainUsers; i++) {
    int numRatings = h_trainUsers[i].size();

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[i * TILE_DEPTH + j].x = h_trainUsers[i][j].first;
      h_ratings[i * TILE_DEPTH + j].y = h_trainUsers[i][j].second * 2;
    }

    h_users[i] = numRatings;

  }
  // move data from host to device
  checkCudaErrors(cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * numRatings, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **) d_users, sizeof(int) * numTrainUsers));
  checkCudaErrors(cudaMemcpy(*d_users, h_users, sizeof(int) * numTrainUsers, cudaMemcpyHostToDevice));

  delete[] h_ratings;
  delete[] h_users;
}

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers) {

  int *d_users;
  Rating *d_allRatings;
  int numTrainUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  float *d_distances;
  short *d_indIdMap;

  cout << "trainUserRatingCount: " << trainUserRatingCount << endl;
  cout << "number of users: " << h_trainUsers.size() << "; effective user: " << numTrainUsers << endl;

  moveRatingsToDevice(h_trainUsers, &d_users, &d_allRatings);
  cout << "data moved to device\n";

  // get global memory
  struct cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, 0));
  cout << "device has " << prop.totalGlobalMem << " global memory\n";

  // calculate how many distances GPU can store, e.g. size of stage
  int ratingsSize = numTrainUsers * TILE_DEPTH * sizeof(Rating);
  int freeMemSize = prop.totalGlobalMem - ratingsSize * 1.5;
  int stageHeight = min(freeMemSize / (numTrainUsers * sizeof(float)) / TILE_SIZE, (long) numTrainUsers / TILE_SIZE);

  // allocate memory for distances
  checkCudaErrors(cudaMalloc((void **) &d_distances, sizeof(float) * numTrainUsers * stageHeight * TILE_SIZE));
  // allocate memory for map(neighborIndex->neighborUserId)
  checkCudaErrors(cudaMalloc((void **) &d_indIdMap, sizeof(unsigned short) * numTrainUsers));

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  cout << "each kernel has " << stageHeight << " blocks\n";
  cout << (numTrainUsers + stageHeight * TILE_SIZE - 1) / (stageHeight * TILE_SIZE) << " kernels will be launched\n";

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  for (int stageStartUser = 0; stageStartUser < numTrainUsers; stageStartUser += stageHeight * TILE_SIZE) {
    int effectiveStageHeight = min(stageHeight, (numTrainUsers - stageStartUser) / TILE_SIZE);
    calculateAllDistance<<<effectiveStageHeight, threadsPerBlock>>>
    (stageStartUser, numTrainUsers, d_users, d_allRatings, d_distances);

    // KNN
    for (int userNum = 0; userNum < stageHeight * TILE_SIZE; userNum++) {
      // sort

      sortNeighbors(d_distances + userNum * numTrainUsers, numTrainUsers, &d_indIdMap);

    }
  }
  for (int x = 0; x < numTrainUsers; x++) {
    cout << d_indIdMap[x] << " ";
  }
  cout << endl;

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout << "kernel ended, took " << milliseconds << "ms\n";

  /* Free memory */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(cudaFree(d_allRatings));
  checkCudaErrors(cudaFree(d_users));
  cudaDeviceReset();
}

