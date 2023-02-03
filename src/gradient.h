#ifndef CORETRAINER_SRC_GRADIENT_H_
#define CORETRAINER_SRC_GRADIENT_H_

#include <cstring>

struct Gradient {

    alignas(64) float *L_0_BIAS_GRADIENT;
    alignas(64) float *L_0_WEIGHT_GRADIENT;

    alignas(64) float L_1_BIAS_GRADIENT = 0;
    alignas(64) float *L_1_WEIGHT_GRADIENT;

    Gradient() {
        L_0_BIAS_GRADIENT = new float[L_1_SIZE];
        L_0_WEIGHT_GRADIENT = new float[L_0_SIZE * L_1_SIZE];
        L_1_WEIGHT_GRADIENT = new float[2 * L_1_SIZE * L_2_SIZE];
    }

    ~Gradient() {
        delete[] L_0_BIAS_GRADIENT;
        delete[] L_0_WEIGHT_GRADIENT;
        delete[] L_1_WEIGHT_GRADIENT;
    }

    inline void reset() {
        L_1_BIAS_GRADIENT = 0;

        std::memset(L_0_BIAS_GRADIENT, 0, L_1_SIZE * sizeof(float));
        std::memset(L_0_WEIGHT_GRADIENT, 0, L_0_SIZE * L_1_SIZE * sizeof(float));
        std::memset(L_1_WEIGHT_GRADIENT, 0, 2 * L_1_SIZE * L_2_SIZE * sizeof(float));
    }
};

#endif //CORETRAINER_SRC_GRADIENT_H_
