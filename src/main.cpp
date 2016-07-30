/*
 * main.cpp
 *
 *  Created on: May 24, 2016
 *      Author: tyler
 */
#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "argparser.h"
#include "common.h"
#include "datareader.h"
#include "cuda_knn.h"

int main(int argc, char ** argv) {
  struct arguments args;
  args.method = 0;
  args.matchedOnly = false;
  args.prettyPrint = false;
  args.maxRating = 5;

  argp_parse(&argp, argc, argv, 0, 0, &args);

  int trainUserRatingCount, testUserRatingCount;
  H_Users h_trainUsers, h_testUsers;
  h_trainUsers.reserve(args.userNum);
  trainUserRatingCount = readData(args.trainFile, h_trainUsers);
  testUserRatingCount = readData(args.testFile, h_testUsers);

  cudaCore(
      trainUserRatingCount,
      testUserRatingCount,
      h_trainUsers,
      h_testUsers,
      args.k);

  return 0;
}
