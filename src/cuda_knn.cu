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
#include "utils.cuh"
#include "thrust_utils.h"

using namespace std;

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
 * CUDA kernel that computes distances between every two users in d_trainUsers
 */
__global__
void calculateAllDistance(
    int stageStartUser,
    int numUsers,
    int *d_trainUsers,
    Rating *allRatings,
    float *d_distances) {

  int globalUserId = stageStartUser + blockIdx.x * blockDim.x + threadIdx.x;
  // user id relative in stage
  int localUserId = blockIdx.x * blockDim.x + threadIdx.x;
  // TODO: experimental, need optimization
  // space for TILE_SIZE * 2 users, each one has at most TILE_DEPTH ratings
  __shared__ Rating ratings[TILE_DEPTH * TILE_SIZE * 2];

  int baseNumRatings = d_trainUsers[globalUserId];
  int numRatings = min(baseNumRatings, TILE_DEPTH);
  Rating *baseStart = ratings + (threadIdx.x + TILE_SIZE) * TILE_DEPTH,
      *baseEnd = baseStart + numRatings;

  // copy data to shared memory, base users are the last TILE_SIZE users in ratings
  Rating *copyFrom = allRatings + globalUserId * TILE_DEPTH;
#pragma unroll
  for (int i = threadIdx.y; i < numRatings; i += TILE_SIZE)
    baseStart[i] = copyFrom[i];
  __syncthreads();

  // TILE_SIZE user per iteration for now
  for (int i = threadIdx.y; i < numUsers; i += TILE_SIZE) {
    int neighborNumRatings = d_trainUsers[i];
    int nbNumRatings = min(neighborNumRatings, TILE_DEPTH);
    Rating *neighborStart = ratings + threadIdx.y * TILE_DEPTH,
        *neighborEnd = neighborStart + nbNumRatings;

    copyFrom = allRatings + i * TILE_DEPTH;
    // copy data to shared memory, neighbors are the first TILE_SIZE users
#pragma unroll
    for (int j = threadIdx.x; j < nbNumRatings; j += TILE_SIZE)
      neighborStart[j] = copyFrom[j];
    __syncthreads();

    d_distances[localUserId * numUsers + i]
                = calculateCOSDistance(baseStart, baseEnd, neighborStart, neighborEnd);

//    if (globalUserId == 766) {
//      printf("distance from user %d to user %d is %.20lf\n", globalUserId, i,
//          d_distances[localUserId * numUsers + i]);
//    }
    __syncthreads();
  }

}

/**
 * CUDA kernel that computes KNN
 */
__global__
void knn(int numUsers, int k,
    short *idxIdMap,
    Rating *trainRatings, Rating *testRatings,
    int *trainUser,
    int *ratingSums, int *ratingCounts) {

  extern __shared__ Rating sharedRatings[];
  // space to store ratings found by each thread
  short *foundRatings = (short*)&sharedRatings[TILE_DEPTH * blockDim.y];
  short *finished = (short*)&foundRatings[blockDim.x * blockDim.y];

  int threadId = threadIdx.x * blockDim.y + threadIdx.y;
  // initialize shared memory
  foundRatings[threadId] = 0;
  if (threadIdx.y == 0 ) finished[threadIdx.x] = 0;

  int sumOfRatings = 0;
  int numOfMatchedNeighbors = 0;

  int testItemId = testRatings[threadIdx.x].x;

  // TODO: consider stopping at 20*K instead of numUsers
  for (int neighborIdx = threadIdx.y; neighborIdx < numUsers; neighborIdx += blockDim.y) {
    if (finished[threadIdx.x]) break;

    // load ratings of blockDim.y users to shared memory
    int neighborNumRatings = trainUser[idxIdMap[neighborIdx]];
    int nbNumRatings = min(neighborNumRatings, TILE_DEPTH);

    Rating *neighborStart = sharedRatings + threadIdx.y * TILE_DEPTH;
    Rating *copyFrom = trainRatings + idxIdMap[neighborIdx] * TILE_DEPTH;

    // TODO: optimize loading by using row major access
    for (int j = threadIdx.x; j < nbNumRatings; j += blockDim.x)
        neighborStart[j] = copyFrom[j];

    foundRatings[threadId] = findItemRating(testItemId, neighborStart, nbNumRatings);
    // thread 0 of each movie collects information
    if (threadIdx.y == 0) {
      for (int i = 0; i < blockDim.y; i++) {
        if (numOfMatchedNeighbors == k) {
          finished[threadIdx.x] = 1;
          break;
        }
        int rate = foundRatings[threadIdx.x * blockDim.y + i];
        if ( rate > 0) {
          sumOfRatings+= rate;
          numOfMatchedNeighbors++;
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.y == 0) {
    ratingSums[threadIdx.x] = sumOfRatings;
    ratingCounts[threadIdx.x] = numOfMatchedNeighbors;
//    printf("prediction for item %d is %f\n", testItemId, (float)sumOfRatings/numOfMatchedNeighbors/2);
  }
}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    int **d_users,
    Rating **d_ratings) {

  // make numTrainUsers a multiple of TILE_SIZE
  int numUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  int numRatings = numUsers * TILE_DEPTH;
  int *h_users = new int[numUsers];

  for (int i = 0; i < numUsers; i++)
    h_users[i] = 0;

  Rating *h_ratings = new Rating[sizeof(Rating) * numRatings];
  checkCudaErrors(cudaMalloc((void **) d_ratings, sizeof(Rating) * numRatings));
  cout << "total size of ratings in bytes: " << sizeof(Rating) * numRatings << endl;

  for (int i = 0; i < numUsers; i++) {
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
  checkCudaErrors(cudaMalloc((void **) d_users, sizeof(int) * numUsers));
  checkCudaErrors(cudaMemcpy(*d_users, h_users, sizeof(int) * numUsers, cudaMemcpyHostToDevice));

  delete[] h_ratings;
  delete[] h_users;
}

void initUsers(User *users, int num) {
  for (int i = 0; i < num; i++)
    users[i] = {NULL, 0};
}

void moveTestRatingsToDevice(
    H_Users h_testUsers,
    User *h_users,
    Rating **d_ratings,
    int testUserRatingCount) {

  int numTestUsers = h_testUsers.size();
  initUsers(h_users, numTestUsers);

  Rating *h_ratings = new Rating[sizeof(Rating) * testUserRatingCount];
  checkCudaErrors(cudaMalloc((void ** )d_ratings, sizeof(Rating) * testUserRatingCount));
  cout << "total size of ratings in bytes: " << sizeof(Rating) * testUserRatingCount << endl;

  int ratingsSoFar = 0;
  for (int i = 0; i < numTestUsers; i++) {
    int numRatings = h_testUsers[i].size();
    if (numRatings < 1) continue;

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[ratingsSoFar + j].x = h_testUsers[i][j].first;
      h_ratings[ratingsSoFar + j].y = h_testUsers[i][j].second * 2;
    }
    // save index
    h_users[i].ratings = *d_ratings + ratingsSoFar;
    h_users[i].numRatings = numRatings;

    ratingsSoFar += numRatings;
  }
  // move data from host to device
  checkCudaErrors(cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * testUserRatingCount, cudaMemcpyHostToDevice));

  delete[] h_ratings;
}


void cudaCore(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers,
    int k) {

  int *d_trainUsers, *d_ratingSums, *d_ratingCounts;
  int h_ratingCounts[32], h_ratingSums[32];
  User *h_testUsersIdx = new User[h_testUsers.size()];
  Rating *d_trainRatings, *d_testRatings;
  int numTrainUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  float *d_distances;
  short *d_idxIdMap;

  int predictedCount = 0;
  float errorSum = 0, errorSumSq = 0;

  cout << "trainUserRatingCount: " << trainUserRatingCount << endl;
  cout << "number of users: " << h_trainUsers.size() << "; effective user: " << numTrainUsers << endl;
  cout << "testUserRatingCount: " << testUserRatingCount << endl;
  cout << "number of test users: " << h_testUsers.size() << endl;

  moveRatingsToDevice(h_trainUsers, &d_trainUsers, &d_trainRatings);
  moveTestRatingsToDevice(h_testUsers, h_testUsersIdx, &d_testRatings, testUserRatingCount);
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

  checkCudaErrors(cudaMalloc((void **) &d_ratingSums, sizeof(int) * 32));
  checkCudaErrors(cudaMalloc((void **) &d_ratingCounts, sizeof(int) * 32));

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
    (stageStartUser, numTrainUsers, d_trainUsers, d_trainRatings, d_distances);

    // KNN
    for (int testUserIdOffset = 0; testUserIdOffset < stageHeight * TILE_SIZE; testUserIdOffset++) {
      int numTestItems = h_testUsersIdx[stageStartUser + testUserIdOffset].numRatings;
      if (numTestItems < 1) continue;

      // sort
      sortNeighbors(d_distances + testUserIdOffset * numTrainUsers, numTrainUsers, &d_idxIdMap);
      // predict
      if (numTestItems > 32) {
//        cout << "more than 32 test items\n";
        continue;
      }
      dim3 threadsPerBlock(numTestItems, 32);
      knn<<<1, threadsPerBlock, 32*TILE_DEPTH*sizeof(Rating) + (numTestItems*33)*sizeof(short)>>>
          (numTrainUsers, k,
          d_idxIdMap,
          d_trainRatings, h_testUsersIdx[stageStartUser + testUserIdOffset].ratings,
          d_trainUsers,
          d_ratingSums, d_ratingCounts);

      cudaDeviceSynchronize();

      checkCudaErrors(cudaMemcpy(h_ratingSums, d_ratingSums, sizeof(int) * 32, cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(h_ratingCounts, d_ratingCounts, sizeof(int) * 32, cudaMemcpyDeviceToHost));

      cudaDeviceSynchronize();
      for (int i = 0; i < numTestItems; i++) {
        short actual = h_testUsers[stageStartUser + testUserIdOffset][i].second;
        if (h_ratingCounts[i] == 0) continue;
        float prediction = h_ratingSums[i] / (float)h_ratingCounts[i] / 2;
        cout << "user: " << stageStartUser + testUserIdOffset << " item: " << h_testUsers[stageStartUser + testUserIdOffset][i].first
            << " actual = " << actual << " predicted = "<< prediction << " based on " << h_ratingCounts[i] << " ratings\n";
        errorSum += fabs(actual - prediction);
        errorSumSq += pow(actual - prediction, 2);
        predictedCount++;
      }
    }
  }
  printptr<<<1,1>>>(d_idxIdMap, numTrainUsers);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout << "kernel ended, took " << milliseconds << "ms\n";

  double mae = errorSum / predictedCount,
    rmse = sqrt(errorSumSq / predictedCount);
  cout << "MAE = " << mae << endl;
  cout << "RMSE = " << rmse << endl;
  cout << "Predicted count  = " << predictedCount << endl;

  cudaDeviceSynchronize();
  /* Free memory */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  checkCudaErrors(cudaFree(d_trainRatings));
  checkCudaErrors(cudaFree(d_testRatings));
  checkCudaErrors(cudaFree(d_trainUsers));
  checkCudaErrors(cudaFree(d_distances));
  checkCudaErrors(cudaFree(d_ratingSums));
  checkCudaErrors(cudaFree(d_ratingCounts));
  cudaDeviceReset();
  delete[] h_testUsersIdx;
}
