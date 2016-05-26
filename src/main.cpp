/*
 * main.cpp
 *
 *  Created on: May 24, 2016
 *      Author: tyler
 */
#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "argparser.hpp"
#include "common.hpp"
#include "datareader.hpp"
#include "cuda_knn.hpp"

int main(int argc, char ** argv) {
  struct arguments args;
  args.method = 0;
  args.matchedOnly = false;
  args.prettyPrint = false;
  args.maxRating = 5;

  argp_parse(&argp, argc, argv, 0, 0, &args);

  int trainUserRatingCount;
  H_Users h_trainUsers(args.userNum), h_testUsers;
  trainUserRatingCount = readData(args.trainFile, h_trainUsers);

  computeAllDistances(
      trainUserRatingCount,
      0,
      h_trainUsers,
      h_testUsers);

  return 0;
}
