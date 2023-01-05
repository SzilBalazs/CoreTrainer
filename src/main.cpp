#include <iostream>
#include <string>

#include "constants.h"
#include "train.h"

int main(int argc, char *argv[]) {

    if (argc != 4 && argc != 5) {
        std::cout << "Train a new network: ./CoreTrainer networkName trainingData validationData" << std::endl;
        std::cout << "Load and train an existing network: ./CoreTrainer networkName trainingData validationData networkPath" << std::endl;
        return 0;
    }

    std::string networkName(argv[1]);
    std::string trainPath(argv[2]);
    std::string validationPath(argv[3]);

    Model NNUE;

    if (argc == 5) {
        std::string networkPath(argv[4]);
        FILE *f = fopen(networkPath.c_str(), "rb");

        if (f == nullptr) {
            std::cout << "Unable to locate existing network!" << std::endl;
            return 0;
        }

        NNUE.loadFromFile(f);
        fclose(f);
        std::cout << networkPath << " successfully loaded!" << std::endl;
    } else {
        std::cout << "No existing network was given. Starting a new network!" << std::endl;
    }

    std::cout << "Network name: " << networkName << "\nTraining data: " << trainPath
              << "\nValidation data: " << validationPath << "\nBatch size: " << BATCH_SIZE
              << "\nLearning rate: " << LR << "\nEval influence: " << EVAL_INFLUENCE
              << "\nThreads: " << THREAD_COUNT << std::endl;

    train(networkName, trainPath, validationPath, NNUE);
    return 0;
}
