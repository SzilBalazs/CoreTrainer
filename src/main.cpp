#include <iostream>
#include <string>

#include "constants.h"
#include "train.h"

void shuffle(const std::string &inPath, const std::string &outPath, int blockCount, int lineCount) {

    std::ifstream in(inPath);
    std::ofstream out(outPath, std::ios::app);

    const int blockSize = lineCount / blockCount;

    std::random_device rd;
    std::mt19937 g(rd());

    for (int block = 1; block <= blockCount; block++) {

        std::vector<std::string> lines;
        std::cout << "\r                                        " << std::flush;
        std::cout << "\rReading block " << block << "..." << std::flush;
        for (int i = 0; i < blockSize; i++) {
            std::string line;
            if (std::getline(in, line)) {
                lines.emplace_back(line);
            } else {
                std::cout << "\nFILE END IS REACHED!\nMaybe try a smaller line count." << std::endl;
            }
        }
        std::cout << "\r                                        " << std::flush;
        std::cout << "\rShuffling block " << block << "..." << std::flush;
        std::shuffle(lines.begin(), lines.end(), g);
        std::cout << "\r                                        " << std::flush;
        std::cout << "\rWriting block " << block << "..." << std::flush;
        for (const std::string &line : lines) {
            out << line << "\n";
        }
        std::cout << "\r                                        " << std::flush;
        std::cout << "\rBlock " << block << " has finished!" << std::flush;
    }

    in.close();
    out.close();
}

int main(int argc, char *argv[]) {

    if (argc != 2 && argc != 4 && argc != 5) {
        std::cout << "Train a new network: ./CoreTrainer networkName trainingData validationData" << std::endl;
        std::cout << "Load and train an existing network: ./CoreTrainer networkName trainingData validationData networkPath" << std::endl;
        std::cout << "Shuffle training data: ./CoreTrainer shuffle" << std::endl;
        return 0;
    }

    if (argc == 2) {

        if (strcmp(argv[1], "shuffle") == 0) {
            std::string inFile, outFile;
            int blockCount, lineCount;
            std::cout << "Input file path: ";
            std::cin >> inFile;
            std::cout << "Output file path: ";
            std::cin >> outFile;
            std::cout << "Block count: ";
            std::cin >> blockCount;
            std::cout << "Line count: ";
            std::cin >> lineCount;

            shuffle(inFile, outFile, blockCount, lineCount);
        } else {
            std::cout << "Unknown parameter: " << argv[1] << std::endl;
        }

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
