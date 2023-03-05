#pragma once

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
    bool stm = BLACK;

    DataEntry() {}

    explicit DataEntry(const std::string &entry);

    constexpr void addFeature(Color color, PieceType type, unsigned int square) {
        whiteFeatureIndexes.push_back((color == BLACK) * 384 + type * 64 + square);
        blackFeatureIndexes.push_back((color == WHITE) * 384 + type * 64 + (square ^ 56));
    }
};

struct Dataset {

    std::ifstream file;

    explicit Dataset(const std::string &fileName);

    bool readEntries(DataEntry *entries, bool &newEpoch);

    inline void close() {
        file.close();
    }
};
