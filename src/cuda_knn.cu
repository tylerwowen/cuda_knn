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

  int globalUserId = stageStartUser + blockIdx.x * TILE_SIZE + threadIdx.x;
  // user id relative in stage
  int localUserId = blockIdx.x * TILE_SIZE + threadIdx.x;
  // TODO: experimental, need optimization
  // space for TILE_SIZE * 2 users, each one has at most TILE_DEPTH ratings
  __shared__ Rating ratings[TILE_DEPTH * TILE_SIZE * 2];

  int numRatings = d_trainUsers[globalUserId];
  Rating *baseStart = ratings + (threadIdx.x + TILE_SIZE) * TILE_DEPTH,
      *baseEnd = baseStart + numRatings;

  // copy data to shared memory, base users are the last TILE_SIZE users in ratings
  Rating *copyFrom = allRatings + globalUserId * TILE_DEPTH;
#pragma unroll
  for (int i = threadIdx.y; i < numRatings; i += TILE_SIZE)
    baseStart[i] = copyFrom[i];
  __syncthreads();

  // TILE_SIZE users per iteration for now
  for (int i = threadIdx.y; i < numUsers; i += TILE_SIZE) {
    int nbNumRatings = d_trainUsers[i];
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

//    if (globalUserId == 0) {
//      printf("%d, %.10lf\n", i+1,
//          d_distances[localUserId * numUsers + i]);
//    }
    __syncthreads();
  }

}

/**
 * CUDA kernel that computes KNN
 */
__global__
void knn_8(int numUsers, int k,
    short *idxIdMap,
    Rating *trainRatings, Rating *testRatings,
    int *trainUser,
    int *ratingSums, int *ratingCounts) {

  // space to store ratings found by each thread
  extern __shared__ short foundRatings[];
  short *finished = (short*) &foundRatings[blockDim.x * NUM_NEIGHBORS];

  int threadId = threadIdx.x * NUM_NEIGHBORS + threadIdx.y;
  // initialize shared memory
  foundRatings[threadId] = 0;
  if (threadIdx.y == 0 ) finished[threadIdx.x] = 0;

  int sumOfRatings = 0;
  int numOfMatchedNeighbors = 0;

  int testItemId = testRatings[threadIdx.x].x;

  // TODO: consider stopping at 20*K instead of numUsers
  for (int neighborIdx = threadIdx.y; neighborIdx < numUsers; neighborIdx += NUM_NEIGHBORS) {
    // load ratings of NUM_NEIGHBORS users to shared memory
    int nbNumRatings = trainUser[idxIdMap[neighborIdx]];

    Rating *neighborStart = trainRatings + idxIdMap[neighborIdx] * TILE_DEPTH;
    __syncthreads();
    if (!finished[threadIdx.x]) {
      foundRatings[threadId] = findItemRating(testItemId, neighborStart, nbNumRatings);
      __syncthreads();

      // thread 0 of each movie collects information
      if (threadIdx.y == 0) {
        int count = min(numUsers - neighborIdx, NUM_NEIGHBORS);
        for (int i = 0; i < count; i++) {
          if (numOfMatchedNeighbors == k) {
            finished[threadIdx.x] = 1;
            break;
          }
          int rate = foundRatings[threadId + i];
          if (rate > 0) {
            sumOfRatings += rate;
            numOfMatchedNeighbors++;
          }
        }
      }
    }
  }
  if (threadIdx.y == 0) {
    ratingSums[threadIdx.x] = sumOfRatings;
    ratingCounts[threadIdx.x] = numOfMatchedNeighbors;
  }
}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    int **d_users,
    Rating **d_ratings) {

  // make numTrainUsers a multiple of TILE_SIZE
  int numUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  int totalNumRatings = numUsers * TILE_DEPTH;
  int *h_users = new int[numUsers];

  for (int i = 0; i < numUsers; i++)
    h_users[i] = 0;

  Rating *h_ratings = new Rating[sizeof(Rating) * totalNumRatings];
  checkCudaErrors(cudaMalloc((void **) d_ratings, sizeof(Rating) * totalNumRatings));
  cout << "size of train ratings in bytes: " << sizeof(Rating) * totalNumRatings << endl;

  for (int i = 0; i < numUsers; i++) {
    int numRatings = min((int)h_trainUsers[i].size(), TILE_DEPTH);

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[i * TILE_DEPTH + j].x = h_trainUsers[i][j].first;
      h_ratings[i * TILE_DEPTH + j].y = h_trainUsers[i][j].second * 2;
    }

    h_users[i] = numRatings;
  }
  // move data from host to device
  checkCudaErrors(cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * totalNumRatings, cudaMemcpyHostToDevice));
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
    int numUsers,
    int testUserRatingCount) {

  initUsers(h_users, numUsers);
  numUsers = min(numUsers, (int) h_testUsers.size());

  Rating *h_ratings = new Rating[sizeof(Rating) * testUserRatingCount];
  checkCudaErrors(cudaMalloc((void **) d_ratings, sizeof(Rating) * testUserRatingCount));

  int ratingsSoFar = 0;
  for (int i = 0; i < numUsers; i++) {
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
  int h_ratingCounts[CONC_ITEMS_NUM] = { 0 }, h_ratingSums[CONC_ITEMS_NUM] = { 0 };
  Rating *d_trainRatings, *d_testRatings;
  int numTrainUsers = h_trainUsers.size() / TILE_SIZE * TILE_SIZE;
  User *h_testUsersIdx = new User[numTrainUsers];
  float *d_distances;
  short *d_idxIdMap;

  int predictedCount = 0;
  double errorSum = 0, errorSumSq = 0;

  cout << "trainUserRatingCount: " << trainUserRatingCount << endl;
  cout << "number of users: " << h_trainUsers.size() << "; effective user: " << numTrainUsers << endl;
  cout << "testUserRatingCount: " << testUserRatingCount << endl;
  cout << "number of test users: " << h_testUsers.size() << endl;

  moveRatingsToDevice(h_trainUsers, &d_trainUsers, &d_trainRatings);
  moveTestRatingsToDevice(h_testUsers, h_testUsersIdx, &d_testRatings, numTrainUsers, testUserRatingCount);
  cout << "data moved to device\n";

  // get free memory
  size_t freeMemSize, totalMemSize;
  checkCudaErrors(cudaMemGetInfo(&freeMemSize, &totalMemSize));
  cout << "device has " << freeMemSize << " free global memory\n";

  checkCudaErrors(cudaMalloc((void **) &d_ratingSums, CONC_ITEMS_NUM * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **) &d_ratingCounts, CONC_ITEMS_NUM * sizeof(int)));
  checkCudaErrors(cudaMalloc((void **) &d_idxIdMap, numTrainUsers * sizeof(short)));

  // calculate how many distances GPU can store, e.g. size of stage
  size_t ratingsSize = numTrainUsers * TILE_DEPTH * sizeof(Rating);
  freeMemSize -= ratingsSize * 10;
  cout << "train rating size " << ratingsSize << "\nfreeMemSize is " << freeMemSize << endl;
  int stageHeight = min(freeMemSize / (numTrainUsers * sizeof(float)) / TILE_SIZE, (long) numTrainUsers / TILE_SIZE);

  // allocate memory for distances
  checkCudaErrors(cudaMalloc((void **) &d_distances, sizeof(float) * numTrainUsers * stageHeight * TILE_SIZE));
  cudaDeviceSynchronize();

  dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
  cout << "each kernel has " << stageHeight << " blocks\n";
  cout << (numTrainUsers + stageHeight * TILE_SIZE - 1) / (stageHeight * TILE_SIZE) << " kernels will be launched\n";

  cudaEvent_t start, stop;
  float milliseconds = 0, distanceCalTime = 0, knnCalTime = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int stageStartUser = 0; stageStartUser < numTrainUsers; stageStartUser += stageHeight * TILE_SIZE) {
    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int effectiveStageHeight = min(stageHeight, (numTrainUsers - stageStartUser) / TILE_SIZE);
    calculateAllDistance<<<effectiveStageHeight, threadsPerBlock>>>
    (stageStartUser, numTrainUsers, d_trainUsers, d_trainRatings, d_distances);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    distanceCalTime += milliseconds;

    // KNN
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    int maxTestUserIdOffset = min(effectiveStageHeight * TILE_SIZE, (int)h_testUsers.size() - stageStartUser);
    for (int testUserIdOffset = 0; testUserIdOffset < maxTestUserIdOffset; testUserIdOffset++) {
      int numTestItems = h_testUsersIdx[stageStartUser + testUserIdOffset].numRatings;
      if (numTestItems < 1) continue;

      // sort
      sortNeighbors(d_distances + testUserIdOffset * numTrainUsers, numTrainUsers, d_idxIdMap);

      // predict
      int numBlocks = (numTestItems + CONC_ITEMS_NUM - 1) / CONC_ITEMS_NUM;
      int remaining = numTestItems;
      for (int block = 0; block < numBlocks; block++) {
        int itemsInBlock = min(remaining, CONC_ITEMS_NUM);
        remaining -= CONC_ITEMS_NUM;
        dim3 threadsPerBlock(itemsInBlock, NUM_NEIGHBORS);

        knn_8<<<1, threadsPerBlock, (itemsInBlock*(NUM_NEIGHBORS+1))*sizeof(short)>>>
        (numTrainUsers, k,
            d_idxIdMap,
            d_trainRatings, h_testUsersIdx[stageStartUser + testUserIdOffset].ratings + block * CONC_ITEMS_NUM,
            d_trainUsers,
            d_ratingSums, d_ratingCounts);

        checkCudaErrors(cudaMemcpy(h_ratingSums, d_ratingSums, sizeof(int) * itemsInBlock, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(h_ratingCounts, d_ratingCounts, sizeof(int) * itemsInBlock, cudaMemcpyDeviceToHost));

        for (int i = 0; i < itemsInBlock; i++) {
          if (h_ratingCounts[i] == 0)
            continue;
          double actual = h_testUsers[stageStartUser + testUserIdOffset][i + block * CONC_ITEMS_NUM].second;
          double prediction = (double)h_ratingSums[i] / 2 /  h_ratingCounts[i];
//          cout << "user: " << stageStartUser + testUserIdOffset + 1
//              << " item: " << h_testUsers[stageStartUser + testUserIdOffset][i+block * itemsInBlock].first
//              << " actual = " << actual << " predicted = "<< prediction << "\n";// " based on " << h_ratingCounts[i] << " ratings\n";
//          if (stageStartUser + testUserIdOffset + 1 < 10) {
//          cout  << stageStartUser + testUserIdOffset + 1
//                        << ", " << h_testUsers[stageStartUser + testUserIdOffset][i+block * CONC_ITEMS_NUM].first
//                        << ", " << actual << ", "<< prediction << ", " << h_ratingSums[i] << ", " << h_ratingCounts[i] << "\n";
//          }
          errorSum += fabs(actual - prediction);
          errorSumSq += pow(actual - prediction, 2);
          predictedCount++;
        }
      }
    }
    cout << "\nerror sum so far: " << errorSum << ", error sum squared so far " << errorSumSq << endl;
    double mae = errorSum / predictedCount,
        rmse = sqrt(errorSumSq / predictedCount);
    cout << "MAE = " << mae << endl;
    cout << "RMSE = " << rmse << endl;
    cout << "Predicted count so far = " << predictedCount << endl;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    knnCalTime += milliseconds;
  }
//  printptr<<<1,1>>>(d_idxIdMap, numTrainUsers);

  cout << "\ndistance calculation took " << distanceCalTime << "ms\n";
  cout << "knn took " << knnCalTime << "ms\n";

  double mae = errorSum / predictedCount,
    rmse = sqrt(errorSumSq / predictedCount);
  cout << "MAE = " << mae << endl;
  cout << "RMSE = " << rmse << endl;
  cout << "Predicted count = " << predictedCount << endl;

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
  checkCudaErrors(cudaFree(d_idxIdMap));
  cudaDeviceReset();
  delete[] h_testUsersIdx;
}
