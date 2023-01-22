#include "dataset.h"
#include "constants.h"
#include <cassert>

DataEntry::DataEntry(const std::string &entry) {
    unsigned int sq = 56, idx = 0;
    bool posSegment = true;
    unsigned int wKing = 64, bKing = 64;
    while (entry[idx] != '<' && idx < entry.size()) {
        char c = entry[idx];

        if (c == ' ') posSegment = false;

        if (posSegment) {
            if ('1' <= c && c <= '8') {
                sq += c - '0';
            } else if (c == '/') {
                sq -= 16;
            } else {
                switch (c) {
                    case 'K':
                        addFeature(WHITE, KING, sq);
                        wKing = sq;
                        break;
                    case 'k':
                        addFeature(BLACK, KING, sq);
                        bKing = sq;
                        break;
                    case 'P':
                        addFeature(WHITE, PAWN, sq);
                        break;
                    case 'p':
                        addFeature(BLACK, PAWN, sq);
                        break;
                    case 'N':
                        addFeature(WHITE, KNIGHT, sq);
                        break;
                    case 'n':
                        addFeature(BLACK, KNIGHT, sq);
                        break;
                    case 'B':
                        addFeature(WHITE, BISHOP, sq);
                        break;
                    case 'b':
                        addFeature(BLACK, BISHOP, sq);
                        break;
                    case 'R':
                        addFeature(WHITE, ROOK, sq);
                        break;
                    case 'r':
                        addFeature(BLACK, ROOK, sq);
                        break;
                    case 'Q':
                        addFeature(WHITE, QUEEN, sq);
                        break;
                    case 'q':
                        addFeature(BLACK, QUEEN, sq);
                        break;
                }
                sq++;
            }
        } else if (entry[idx] == 'w')
            stm = WHITE;
        idx++;
    }
    assert(wKing != 64 && bKing != 64);

    // Apply king buckets
    const unsigned int whiteOffset = KING_BUCKET[wKing] * 768;
    const unsigned int blackOffset = KING_BUCKET[bKing ^ 56] * 768;

    for (unsigned int &feature : whiteFeatureIndexes) {
        feature += whiteOffset;
    }

    for (unsigned int &feature : blackFeatureIndexes) {
        feature += blackOffset;
    }

    std::string evalStr;
    for (idx++; entry[idx] != '>' && idx < entry.size(); idx++) {
        evalStr += entry[idx];
    }
    idx++;
    eval = sigmoid(std::stoi(evalStr) / EVAL_SCALE);

    wdl = (entry[idx] == '-') ? 0.0f : (entry[idx] == '0' ? 0.5f : 1.0f);
}

Dataset::Dataset(const std::string &fileName) {
    file = std::ifstream(fileName);

    if (file.fail()) {
        std::cout << "Unable to open " << fileName << "!" << std::endl;
        exit(0);
    }
}

bool Dataset::readEntries(DataEntry *entries, bool &newEpoch) {
    std::string line;
    for (unsigned int idx = 0; idx < BATCH_SIZE; idx++) {
        if (std::getline(file, line)) {
            entries[idx] = DataEntry(line);
        } else {
            idx--;
            file.clear();
            file.seekg(0);
            newEpoch = true;
        }
    }
    return newEpoch;
}