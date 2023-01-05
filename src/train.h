#ifndef CORETRAINER_SRC_TRAIN_H_
#define CORETRAINER_SRC_TRAIN_H_

#include "model.h"

void train(const std::string &networkName, const std::string &trainPath, const std::string &validationPath, Model &model);

#endif //CORETRAINER_SRC_TRAIN_H_
