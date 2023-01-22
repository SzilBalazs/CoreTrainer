#ifndef CORETRAINER_SRC_MODEL_H_
#define CORETRAINER_SRC_MODEL_H_

#include <cmath>

#include "constants.h"
#include "dataset.h"
#include "gradient.h"
#include "layer.h"

struct Model {
    LinearLayer<L_0_SIZE, L_1_SIZE> L_0;
    LinearLayer<2 * L_1_SIZE, L_2_SIZE> L_1;

    inline Model() : L_0(), L_1() {}

    inline void loadFromFile(FILE *f) {
        L_0.loadFromFile(f);
        L_1.loadFromFile(f);
    }

    inline void writeToFile(FILE *f) {
        L_0.writeToFile(f);
        L_1.writeToFile(f);
    }

    inline void exportToFile(FILE *f) {

        float maxWeight0 = 0.0;
        float maxBias0 = 0.0;
        float maxWeight1 = 0.0;
        float maxBias1 = 0.0;

        for (float w : L_0.weights) {
            maxWeight0 = std::max(maxWeight0, w);
            int16_t a = std::round(w * QUANT_SCALE);
            fwrite(&a, sizeof(int16_t), 1, f);
        }

        for (float w : L_0.biases) {
            maxBias0 = std::max(maxBias0, w);
            int16_t a = std::round(w * QUANT_SCALE);
            fwrite(&a, sizeof(int16_t), 1, f);
        }

        for (float w : L_1.weights) {
            maxWeight1 = std::max(maxWeight1, w);
            int16_t a = std::round(w * QUANT_SCALE);
            fwrite(&a, sizeof(int16_t), 1, f);
        }

        for (float w : L_1.biases) {
            maxBias1 = std::max(maxBias1, w);
            int16_t a = std::round(w * QUANT_SCALE * QUANT_SCALE);
            fwrite(&a, sizeof(int16_t), 1, f);
        }

        std::cout << "Max weight: " << maxWeight0 << " " << maxWeight1 << "\nMax bias: " << maxBias0 << " " << maxBias1 << std::endl;
    }

    inline float forward(DataEntry &input, float *hiddenLayer) {
        float output;

        if (input.stm == WHITE) {
            L_0.forward(input.whiteFeatureIndexes, hiddenLayer);
            L_0.forward(input.blackFeatureIndexes, hiddenLayer + L_1_SIZE);
        } else {
            L_0.forward(input.blackFeatureIndexes, hiddenLayer);
            L_0.forward(input.whiteFeatureIndexes, hiddenLayer + L_1_SIZE);
        }

        for (unsigned int idx = 0; idx < 2 * L_1_SIZE; idx++) {
            hiddenLayer[idx] = ReLU(hiddenLayer[idx]);
        }

        L_1.forward(hiddenLayer, &output);
        return output;
    }
};

#endif //CORETRAINER_SRC_MODEL_H_
