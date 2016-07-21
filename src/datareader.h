/*
 * datareader.h
 *
 *  Created on: May 19, 2016
 *      Author: tyler
 */

#ifndef DATAREADER_H_
#define DATAREADER_H_

#include <fstream>
#include <iostream>

#include "common.h"

int readData(std::string const &filename, H_Users &users);

#endif /* DATAREADER_H_ */
