/*
 * datareader.hpp
 *
 *  Created on: May 19, 2016
 *      Author: tyler
 */

#ifndef DATAREADER_HPP_
#define DATAREADER_HPP_

#include <fstream>
#include <iostream>

#include "common.hpp"

int readData(std::string const &filename, H_Users &users);

#endif /* DATAREADER_HPP_ */
