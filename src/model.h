#ifndef CORETRAINER_SRC_MODEL_H_
#define CORETRAINER_SRC_MODEL_H_

#include "constants.h"
#include "dataset.h"
#include "layer.h"

struct Model {
    LinearLayer<L_0_SIZE, L_1_SIZE> L_0;
    LinearLayer<L_1_SIZE, L_2_SIZE> L_1;

    inline Model() : L_0(), L_1() {}

    inline void loadFromFile(FILE *f) {
        L_0.loadFromFile(f);
        L_1.loadFromFile(f);
    }

    inline void writeToFile(FILE *f) {
        L_0.writeToFile(f);
        L_1.writeToFile(f);
    }

    inline float forward(DataEntry &input, float *hiddenLayer) {
        float output;

        L_0.forward(input.whiteFeatureIndexes, hiddenLayer);

        for (unsigned int idx = 0; idx < L_1_SIZE; idx++) {
            hiddenLayer[idx] = ReLU(hiddenLayer[idx]);
        }

        L_1.forward(hiddenLayer, &output);

        return output;
    }
};

#endif //CORETRAINER_SRC_MODEL_H_
