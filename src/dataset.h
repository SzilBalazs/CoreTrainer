#ifndef CORETRAINER_SRC_DATASET_H_
#define CORETRAINER_SRC_DATASET_H_

#include "constants.h"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct DataEntry {

    std::vector<unsigned int> whiteFeatureIndexes;
    std::vector<unsigned int> blackFeatureIndexes;

    float wdl;
    float eval;
    float expected;
    bool stm = BLACK;

    DataEntry() {}

    explicit DataEntry(const std::string &entry);

    inline void addFeature(Color color, PieceType type, unsigned int square) {
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
