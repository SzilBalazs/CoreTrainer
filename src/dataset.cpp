#include "dataset.h"
#include "constants.h"

DataEntry::DataEntry(const std::string &entry) {
    unsigned int sq = 56, idx = 0;
    bool posSegment = true;

    while (entry[idx] != '[' && idx < entry.size()) {
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
                        break;
                    case 'k':
                        addFeature(BLACK, KING, sq);
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
            stm = false;
        idx++;
    }

    idx++;
    wdl = entry[idx] == '1' ? 1 : (entry[idx + 2] == '5' ? float(0.5) : 0);
    eval = sigmoid(float(std::stoi(entry.substr(idx + 5, entry.size() - idx - 4))));
}

Dataset::Dataset(const std::string &fileName) {
    file = std::ifstream(fileName);

    if (file.fail()) {
        std::cout << "Unable to open " << fileName << "!" << std::endl;
        exit(0);
    }
}

bool Dataset::readEntries(DataEntry *entries) {
    bool newEpoch = false;
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