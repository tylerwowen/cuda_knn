/*
 * datareader.c
 *
 *  Created on: May 19, 2016
 *      Author: tyler
 */
#include "datareader.hpp"

using namespace std;

int readData(string const &filename, H_Users &users) {
  ifstream ifs(filename);
  int ratingCount = 0;
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
    }
    H_Ratings *ratings = &users[userId-1];
    ratings->emplace_back(itemId, rating);
    ratingCount++;
  }

  return ratingCount;
}
