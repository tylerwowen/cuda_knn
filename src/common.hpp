/*
 * common.hpp
 *
 *  Created on: May 18, 2016
 *      Author: tyler
 */

#ifndef COMMON_HPP_
#define COMMON_HPP_

#include <vector>

#define COS 0
#define L1 1
#define L2 2
#define PCC 3
#define LLR 4

/**
 *  {itemId, rating}
 */
typedef std::pair<int, float> H_Rating;

/**
 *  [{itemId, rating}, ...]
 */
typedef std::vector<H_Rating> H_Ratings;

/**
 *  [{userId, Ratings}, ...]
 */
typedef std::vector<H_Ratings> H_Users;

#endif /* COMMON_HPP_ */
