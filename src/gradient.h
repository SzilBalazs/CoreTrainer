#ifndef CORETRAINER_SRC_GRADIENT_H_
#define CORETRAINER_SRC_GRADIENT_H_

#include <cstring>

struct Gradient {

    alignas(64) float L_0_BIAS_GRADIENT[L_1_SIZE] = {0};
    alignas(64) float L_0_WEIGHT_GRADIENT[L_0_SIZE * L_1_SIZE] = {0};

    alignas(64) float L_1_BIAS_GRADIENT = 0;
    alignas(64) float L_1_WEIGHT_GRADIENT[L_1_SIZE * L_2_SIZE] = {0};

    inline void reset() {
        L_1_BIAS_GRADIENT = 0;

        std::memset(L_0_BIAS_GRADIENT, 0, sizeof(L_0_BIAS_GRADIENT));
        std::memset(L_0_WEIGHT_GRADIENT, 0, sizeof(L_0_WEIGHT_GRADIENT));
        std::memset(L_1_WEIGHT_GRADIENT, 0, sizeof(L_1_WEIGHT_GRADIENT));
    }
};

#endif //CORETRAINER_SRC_GRADIENT_H_
