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
#include "common.hpp"

using namespace std;

#define TILE_SIZE 32
#define TILE_DEPTH 96

__device__
double calculateCOSDistance(Rating *r1Start, Rating *r1End, Rating *r2Start, Rating *r2End) {
  double dotProduct = 0.0f, r1NormSQ = 0.0f, r2NormSQ = 0.0f;

  while (r1Start < r1End && r2Start < r2End) {
    if (r1Start->itemId > r2Start->itemId) {
      // treat r1Start->rating as 0
      r2NormSQ += r2Start->rating * r2Start->rating;
      r2Start++;
    }
    else if (r1Start->itemId == r2Start->itemId) {
      dotProduct += r1Start->rating * r2Start->rating;
      r1NormSQ += r1Start->rating * r1Start->rating;
      r2NormSQ += r2Start->rating * r2Start->rating;
      r1Start++;
      r2Start++;
    }
    else {
      // treat r2Start->rating as 0
      r1NormSQ += r1Start->rating * r1Start->rating;
      r1Start++;
    }
  }
  // finish baseUser tail, if any
  while (r1Start < r1End) {
    r1NormSQ += r1Start->rating * r1Start->rating;
    r1Start++;
  }
  // finish neighbor tail, if any
  while (r2Start < r2End) {
    r2NormSQ += r2Start->rating * r2Start->rating;
    r2Start++;
  }
  // distance
  return  dotProduct / (sqrt(r1NormSQ) * sqrt(r2NormSQ));
}

/**
 * CUDA kernel that computes distances between every two users in d_users
 */
__global__
void calculateAllDistance(int *d_users, int numUsers, Rating *allRatings) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: experimental, need optimization
  // space for TILE_SIZE * 2 users, each one has at most TILE_DEPTH ratings
  __shared__ Rating ratings[TILE_DEPTH * TILE_SIZE * 2];

  int baseNumRatings = d_users[gtid];
  int numRatings = min(baseNumRatings, TILE_DEPTH);
  Rating *baseStart = ratings + (threadIdx.x + TILE_SIZE) * TILE_DEPTH,
      *baseEnd = baseStart + numRatings;

  // copy data to shared memory, base users are the last TILE_SIZE users
  Rating *copyFrom = allRatings + gtid * TILE_DEPTH;
  #pragma unroll
  for (int i = threadIdx.y; i < TILE_DEPTH; i += TILE_DEPTH/TILE_SIZE)
    baseStart[i] = copyFrom[i];
  __syncthreads();

//  printf("hello from block %d thread x %d, thread y %d\n", blockIdx.x, threadIdx.x, threadIdx.y);

  int *tileStartUser = d_users;
  // TILE_SIZE user per time for now
  for (int i = 0; i < numUsers - TILE_SIZE; i += TILE_SIZE, tileStartUser += TILE_SIZE) {
    int neighborNumRatings = tileStartUser[threadIdx.y];
    int nbNumRatings = min(neighborNumRatings, TILE_DEPTH);
    Rating *neighborStart = ratings + threadIdx.y * TILE_DEPTH,
        *neighborEnd = neighborStart + nbNumRatings;

    copyFrom = allRatings + (i + threadIdx.y) * TILE_DEPTH;
    // copy data to shared memory, neighbors are the first TILE_SIZE users
    #pragma unroll
    for (int i = threadIdx.x; i < TILE_DEPTH; i += TILE_DEPTH/TILE_SIZE)
      neighborStart[i] = copyFrom[i];
    // TODO: what if there are more than TILE_DEPTH users
    __syncthreads();

    double distance = calculateCOSDistance(baseStart, baseEnd, neighborStart, neighborEnd);
    __syncthreads();
  }

  //  printf("distance from user %d to user %d is %.20lf\n", userId + 1, i, distance);
}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    int **d_users,
    Rating **d_ratings) {

  int numTrainUsers = h_trainUsers.size();
  int numRatings = numTrainUsers * 192;
  int *h_users =  new int[numTrainUsers];;
  for (int i = 0; i < numTrainUsers; i++)
    h_users[i] = 0;

  Rating *h_ratings = new Rating[sizeof(Rating) * numRatings];
  checkCudaErrors(cudaMalloc((void ** )d_ratings, sizeof(Rating) * numRatings));
  cout << "total size of ratings in bytes: " << sizeof(Rating) * numRatings << endl;

  for (int i = 0; i < numTrainUsers; i++) {
    int numRatings = h_trainUsers[i].size();

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[i * 192 + j].itemId = h_trainUsers[i][j].first;
      h_ratings[i * 192 + j].rating = h_trainUsers[i][j].second;
    }

    h_users[i] = numRatings;

  }
  // move data from host to device
  checkCudaErrors(cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * numRatings, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void ** )d_users, sizeof(int) * numTrainUsers));
  checkCudaErrors(cudaMemcpy(*d_users, h_users, sizeof(int) * numTrainUsers,cudaMemcpyHostToDevice));

  delete[] h_ratings;
  delete[] h_users;
}

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers)
{

  int *d_users;
  Rating *d_allRatings;

  cout << "trainUserRatingCount: " << trainUserRatingCount << endl;
  cout << "number of users: " << h_trainUsers.size() << endl;

  moveRatingsToDevice(h_trainUsers, &d_users, &d_allRatings);
  cout << "data moved to device\n";

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  int numBlocks = h_trainUsers.size() / TILE_SIZE;
  cout << "kernel starts with " << numBlocks << " blocks\n";

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  calculateAllDistance<<<numBlocks, threadsPerBlock>>> (d_users, h_trainUsers.size(), d_allRatings);

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

