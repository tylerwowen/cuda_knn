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

/**
 * CUDA kernel that computes reciprocal values for a given vector
 */
__global__ void calculateDistance(User *d_users, int numUsers) {
  unsigned userId = threadIdx.x; //+ blockIdx.x * 200;
  //  unsigned movieId = threadIdx.x;
  User *baseUser = d_users + userId;
  for (int i = 0; i < numUsers; i++) {
    double dotProduct = 0, baseNormSQ = 0, neighborNormSQ = 0;
    User *neighbor = d_users + i;
    Rating *baseRating = baseUser->ratings,
        *baseEnd = baseUser->ratings + baseUser->numRatings,
        *neighborRating = neighbor->ratings,
        *neighborEnd = neighbor->ratings + neighbor->numRatings;
    while (baseRating < baseEnd && neighborRating < neighborEnd) {
      if (baseRating->itemId == neighborRating->itemId) {
        dotProduct += baseRating->rating * neighborRating->rating;
        baseNormSQ += baseRating->rating * baseRating->rating;
        neighborNormSQ += neighborRating->rating * neighborRating->rating;
        baseRating++;
        neighborRating++;
      } else if (baseRating->itemId > neighborRating->itemId) {
        // treat baseRating->rating as 0
        neighborNormSQ += neighborRating->rating * neighborRating->rating;
        neighborRating++;
      } else {
        // treat neighborRating->rating as 0
        baseNormSQ += baseRating->rating * baseRating->rating;
        baseRating++;
      }
    }
    // finish baseUser tail, if any
    while (baseRating < baseEnd) {
      baseNormSQ += baseRating->rating * baseRating->rating;
      baseRating++;
    }
    // finish neighbor tail, if any
    while (neighborRating < neighborEnd) {
      neighborNormSQ += neighborRating->rating * neighborRating->rating;
      neighborRating++;
    }

    // distance
    double distance = dotProduct / (sqrt(baseNormSQ) * sqrt(neighborNormSQ));
//    printf("distance from user %d to user %d is %.20lf\n", userId + 1, i, distance);
  }
}

void initUsers(User *users, int num) {
  for (int i = 0; i < num; num++)
    users[i] = {NULL, 0};
}

void moveRatingsToDevice(
    H_Users h_trainUsers,
    User **d_users,
    Rating **d_allRatings,
    int trainUserRatingCount) {

  int numTrainUsers = h_trainUsers.size();
  User *h_users =  new User[numTrainUsers];;
  initUsers(h_users, numTrainUsers);

  // TODO: size may not be big enough
  Rating *h_ratings = new Rating[1000], *d_ratings;
  CUDA_CHECK_RETURN(
        cudaMalloc((void ** )d_allRatings,
            sizeof(Rating) * trainUserRatingCount));
  cout << "total size: " << sizeof(Rating) * trainUserRatingCount << endl;

  int ratingsSoFar = 0;
  for (int i = 0; i < numTrainUsers; i++) {
    int numRatings = h_trainUsers[i].size();

    // copy vector to intermediate host array
    for (int j = 0; j < numRatings; j++) {
      h_ratings[j].itemId = h_trainUsers[i][j].first;
      h_ratings[j].rating = h_trainUsers[i][j].second;
    }

    // move data from host to device
    d_ratings = *d_allRatings + ratingsSoFar;
    CUDA_CHECK_RETURN(
        cudaMemcpy(d_ratings, h_ratings, sizeof(Rating) * numRatings,
            cudaMemcpyHostToDevice));

    // save device address
    h_users[i].ratings = d_ratings;
    h_users[i].numRatings = numRatings;

    ratingsSoFar += numRatings;
  }
  CUDA_CHECK_RETURN(cudaMalloc((void ** )d_users, sizeof(User) * (numTrainUsers)));
  CUDA_CHECK_RETURN(cudaMemcpy(*d_users, h_users, sizeof(User) * (numTrainUsers),cudaMemcpyHostToDevice));

  delete[] h_ratings;
  delete[] h_users;
}

void computeAllDistances(
    int trainUserRatingCount,
    int testUserRatingCount,
    H_Users h_trainUsers,
    H_Users h_testUsers)
{

  User *d_users = NULL;
  Rating *d_allRatings = NULL;

  moveRatingsToDevice(h_trainUsers, &d_users, &d_allRatings, trainUserRatingCount);

  cudaEvent_t start, stop;
  float milliseconds = 0;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  calculateDistance<<<1, 942>>> (d_users, h_trainUsers.size());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
  cout << "distance calculation costs " << milliseconds << "ms\n";

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

