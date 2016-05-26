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

static void CheckCudaErrorAux (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

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
 * Calculate all pair-wise distances in a tile
 */
__device__
void tileCalculation(Rating sharedRatings[],
    User *tileStartUser,
    Rating *baseStart,
    Rating *baseEnd) {

  // total 10 threads in x direction
  User *neighbor = tileStartUser + threadIdx.y;
  Rating *neighborStart = sharedRatings + threadIdx.y * 256,
      *neighborEnd = neighborStart + min(neighbor->numRatings, 256);

  // copy data to shared memory, neighbors are the first 10 users
  if (threadIdx.y == 0) {
    for (int i = 0; i < neighbor->numRatings && i < 256; i++)
      neighborStart[i] = neighbor->ratings[i];
    // TODO: what if there are more than 256 users
  }
  __syncthreads();

  double distance = calculateCOSDistance(baseStart, baseEnd, neighborStart, neighborEnd);
//  if (threadIdx.x == 0 && threadIdx.y == 0)
//    printf("distance from 0 to %d is %lf\n", threadIdx.y, distance);
}

/**
 * CUDA kernel that computes distances between every two users in d_users
 */
__global__ void calculateAllDistance(User *d_users, int numUsers) {
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;

  // TODO: experimental, need optimization
  // space for 20 user, each one has at most 256 ratings
  __shared__ Rating ratings[5120];
  User *baseUser = d_users + gtid;
  Rating *ratingStart = ratings + (threadIdx.x + 10)* 256,
      *ratingEnd = ratingStart + min(baseUser->numRatings, 256);

  // copy data to shared memory, base users are the last 10 users
  if (threadIdx.x == 0) {
    for (int i = 0; i < baseUser->numRatings && i < 256; i++)
      ratingStart[i] = baseUser->ratings[i];
    // TODO: what if there are more than 256 users
  }
  __syncthreads();

  User *tileStartUser = d_users;
  // 10 user per time for now
  for (int i = 0; i < numUsers - 10; i += 10) {
    tileCalculation(ratings, tileStartUser, ratingStart, ratingEnd);
    tileStartUser += 10;
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
  CUDA_CHECK_RETURN(
        cudaMalloc((void ** )d_ratings,
            sizeof(Rating) * trainUserRatingCount));
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
  CUDA_CHECK_RETURN(
      cudaMemcpy(*d_ratings, h_ratings, sizeof(Rating) * trainUserRatingCount,
          cudaMemcpyHostToDevice));
  CUDA_CHECK_RETURN(cudaMalloc((void ** )d_users, sizeof(User) * numTrainUsers));
  CUDA_CHECK_RETURN(cudaMemcpy(*d_users, h_users, sizeof(User) * numTrainUsers,cudaMemcpyHostToDevice));

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

  dim3 threadsPerBlock(10, 10);
  cout << "kernel starts\n";

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  calculateAllDistance<<<8, threadsPerBlock>>> (d_users, h_trainUsers.size());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout << "kernel ended, took " << milliseconds << "ms\n";

  /* Free memory */
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  CUDA_CHECK_RETURN(cudaFree(d_allRatings));
  CUDA_CHECK_RETURN(cudaFree(d_users));
  cudaDeviceReset();
}

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux (const char *file, unsigned line, const char *statement, cudaError_t err)
{
  if (err == cudaSuccess)
    return;
  std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
  exit (1);
}

