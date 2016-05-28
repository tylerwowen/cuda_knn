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
void calculateAllDistance(User *d_users, int numUsers) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: experimental, need optimization
  // space for TILE_SIZE * 2 users, each one has at most TILE_DEPTH ratings
  __shared__ Rating ratings[TILE_DEPTH * TILE_SIZE * 2];

  User* baseUser = d_users + gtid;
  int numRatings = min(baseUser->numRatings, TILE_DEPTH);
  Rating *baseStart = ratings + (threadIdx.x + TILE_SIZE) * TILE_DEPTH,
      *baseEnd = baseStart + numRatings;
  if (threadIdx.y == 0) {
    // copy data to shared memory, base users are the last TILE_SIZE users
    for (int i = 0; i < numRatings; i++)
      baseStart[i] = baseUser->ratings[i];
    // TODO: what if there are more than TILE_DEPTH users
  }
  __syncthreads();
//  printf("hello from block %d thread x %d, thread y %d\n", blockIdx.x, threadIdx.x, threadIdx.y);

  User *tileStartUser = d_users;
  // TILE_SIZE user per time for now
  for (int i = 0; i < numUsers - TILE_SIZE; i += TILE_SIZE) {
    User *neighbor = tileStartUser + threadIdx.y;
    int nbNumRatings = min(neighbor->numRatings, TILE_DEPTH);
    Rating *neighborStart = ratings + threadIdx.y * TILE_DEPTH,
        *neighborEnd = neighborStart + nbNumRatings;

    if (threadIdx.x == 0) {
      // copy data to shared memory, neighbors are the first TILE_SIZE users
      for (int i = 0; i < nbNumRatings; i++)
        neighborStart[i] = neighbor->ratings[i];
      // TODO: what if there are more than TILE_DEPTH users
    }
    __syncthreads();

    double distance = calculateCOSDistance(baseStart, baseEnd, neighborStart, neighborEnd);
    tileStartUser += TILE_SIZE;
    __syncthreads();
  }

  //  printf("distance from user %d to user %d is %.20lf\n", userId + 1, i, distance);
}

void initUsers(User *users, int num) {
  for (int i = 0; i < num; num++)
    users[i] = {NULL, 0};
}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    User **d_users,
    Rating **d_ratings,
    int trainUserRatingCount) {

  int numTrainUsers = h_trainUsers.size();
  User *h_users =  new User[numTrainUsers];;
  initUsers(h_users, numTrainUsers);

  Rating *h_ratings = new Rating[sizeof(Rating) * trainUserRatingCount];
  checkCudaErrors(cudaMalloc((void ** )d_ratings, sizeof(Rating) * trainUserRatingCount));
  cout << "total size of ratings in bytes: " << sizeof(Rating) * trainUserRatingCount << endl;

  int ratingsSoFar = 0;
  for (int i = 0; i < numTrainUsers; i++) {
    int numRatings = h_trainUsers[i].size();

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[ratingsSoFar + j].itemId = h_trainUsers[i][j].first;
      h_ratings[ratingsSoFar + j].rating = h_trainUsers[i][j].second;
    }
    // save device address
    h_users[i].ratings = *d_ratings + ratingsSoFar;
    h_users[i].numRatings = numRatings;

    ratingsSoFar += numRatings;
  }
  // move data from host to device
  checkCudaErrors(cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * trainUserRatingCount, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void ** )d_users, sizeof(User) * numTrainUsers));
  checkCudaErrors(cudaMemcpy(*d_users, h_users, sizeof(User) * numTrainUsers,cudaMemcpyHostToDevice));

  delete[] h_ratings;
  delete[] h_users;
}

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers)
{

  User *d_users;
  Rating *d_allRatings;

  cout << "trainUserRatingCount: " << trainUserRatingCount << endl;
  cout << "number of users: " << h_trainUsers.size() << endl;

  moveRatingsToDevice(h_trainUsers, &d_users, &d_allRatings, trainUserRatingCount);
  cout << "data moved to device\n";

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  int numBlocks = h_trainUsers.size() / TILE_SIZE;
  cout << "kernel starts with " << numBlocks << " blocks\n";

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  calculateAllDistance<<<numBlocks, threadsPerBlock>>> (d_users, h_trainUsers.size());

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

