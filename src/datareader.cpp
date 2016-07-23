/*
 * datareader.c
 *
 *  Created on: May 19, 2016
 *      Author: tyler
 */
#include "datareader.h"

using namespace std;

int readData(string const &filename, H_Users &users) {
  ifstream ifs(filename);
  int ratingCount = 0, userRatingCount = 0;
  int lastUser = 0;
  while (true) {
    unsigned int userId, itemId, timestamp;
    float rating;
    ifs >>  userId >> itemId >> rating >> timestamp;
    if (ifs.fail()) {
      break;
    }
    while (userId - lastUser > 0) {
      users.push_back(H_Ratings());
      lastUser++;
      userRatingCount = 0;
    }

    if (userRatingCount >= TILE_DEPTH) continue;

    H_Ratings *ratings = &users[userId-1];
    ratings->emplace_back(itemId, rating);
    ratingCount++;
    userRatingCount++;
  }

  return ratingCount;
}
