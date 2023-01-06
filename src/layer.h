#ifndef CORETRAINER_SRC_LAYER_H_
#define CORETRAINER_SRC_LAYER_H_

#include <cstring>
#include <immintrin.h>
#include <random>

template<unsigned int IN, unsigned int OUT>
struct LinearLayer {
    alignas(64) float biases[OUT] = {0};
    alignas(64) float weights[IN * OUT] = {0};

    LinearLayer() {
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<float> dist(0, 0.1);
        for (unsigned int idx = 0; idx < IN * OUT; idx++) {
            weights[idx] = dist(rng);
        }
    }

    explicit inline LinearLayer(FILE *f) {
        loadFromFile(f);
    }

    constexpr float getWeight(int in, int out) {
        return weights[in * OUT + out];
    }

    inline void loadFromFile(FILE *f) {
        fread(weights, sizeof(float), IN * OUT, f);
        fread(biases, sizeof(float), OUT, f);
    }

    inline void writeToFile(FILE *f) {
        fwrite(weights, sizeof(float), IN * OUT, f);
        fwrite(biases, sizeof(float), OUT, f);
    }

    inline void forward(float *inputLayer, float *outputLayer) {
        memcpy(outputLayer, biases, sizeof(biases));

        for (unsigned int i = 0; i < IN; i++) {
            for (unsigned int j = 0; j < OUT; j++) {
                outputLayer[j] += inputLayer[i] * weights[i * OUT + j];
            }
        }
    }

    inline void forward(const std::vector<unsigned int> &featureIndexes, float *outputLayer) {
        memcpy(outputLayer, biases, sizeof(biases));

        for (unsigned int i : featureIndexes) {
            for (unsigned int j = 0; j < OUT; j++) {
                outputLayer[j] += weights[i * OUT + j];
            }
        }
    }
};

#endif //CORETRAINER_SRC_LAYER_H_
