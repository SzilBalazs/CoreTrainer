#pragma once

#include <cstring>
#include <memory>

struct Gradient {

    std::unique_ptr<float[]> L_0_BIAS_GRADIENT;
    std::unique_ptr<float[]> L_0_WEIGHT_GRADIENT;

    alignas(64) float L_1_BIAS_GRADIENT = 0;
    std::unique_ptr<float[]> L_1_WEIGHT_GRADIENT;

    Gradient() {
        L_0_BIAS_GRADIENT = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(64, L_1_SIZE * sizeof(float))));
        L_0_WEIGHT_GRADIENT = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(64, L_0_SIZE * L_1_SIZE * sizeof(float))));
        L_1_WEIGHT_GRADIENT = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(64, 2 * L_1_SIZE * L_2_SIZE * sizeof(float))));
    }

    inline void reset() {
        L_1_BIAS_GRADIENT = 0;

        std::memset(L_0_BIAS_GRADIENT.get(), 0, L_1_SIZE * sizeof(float));
        std::memset(L_0_WEIGHT_GRADIENT.get(), 0, L_0_SIZE * L_1_SIZE * sizeof(float));
        std::memset(L_1_WEIGHT_GRADIENT.get(), 0, 2 * L_1_SIZE * L_2_SIZE * sizeof(float));
    }
};
