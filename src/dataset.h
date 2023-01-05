#ifndef CORETRAINER_SRC_DATASET_H_
#define CORETRAINER_SRC_DATASET_H_

#include "constants.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct DataEntry {

    float whiteFeatures[L_0_SIZE] = {0};
    float blackFeatures[L_0_SIZE] = {0};

    std::vector<unsigned int> whiteFeatureIndexes;
    std::vector<unsigned int> blackFeatureIndexes;

    float wdl;
    float eval;
    bool stm = true; // false=WHITE true=BLACK

    DataEntry() {}

    explicit DataEntry(const std::string &entry);

    inline void addFeature(Color color, PieceType type, unsigned int square) {
        whiteFeatures[(color == BLACK) * 384 + type * 64 + square] = 1;
        blackFeatures[(color == WHITE) * 384 + type * 64 + (square ^ 56)] = 1;

        whiteFeatureIndexes.push_back((color == BLACK) * 384 + type * 64 + square);
        blackFeatureIndexes.push_back((color == WHITE) * 384 + type * 64 + (square ^ 56));
    }
};

struct Dataset {

    std::ifstream file;

    explicit Dataset(const std::string &fileName);

    bool readEntries(DataEntry *entries);

    inline void close() {
        file.close();
    }
};

#endif //CORETRAINER_SRC_DATASET_H_
