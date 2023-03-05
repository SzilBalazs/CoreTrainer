#pragma once

#include <cstring>
#include <immintrin.h>
#include <random>

template<unsigned int IN, unsigned int OUT>
struct LinearLayer {
    std::unique_ptr<float[]> biases;
    std::unique_ptr<float[]> weights;

    LinearLayer() {

        if constexpr ((OUT * sizeof(float)) % 64 == 0)
            biases = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(64, OUT * sizeof(float))));
        else
            biases = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(4, OUT * sizeof(float))));

        weights = std::unique_ptr<float[]>(static_cast<float *>(aligned_alloc(64, IN * OUT * sizeof(float))));

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
        fread(weights.get(), sizeof(float), IN * OUT, f);
        fread(biases.get(), sizeof(float), OUT, f);
    }

    inline void writeToFile(FILE *f) {
        fwrite(weights.get(), sizeof(float), IN * OUT, f);
        fwrite(biases.get(), sizeof(float), OUT, f);
    }

    inline void forward(float *inputLayer, float *outputLayer) {
        memcpy(outputLayer, biases.get(), OUT * sizeof(float));

        for (unsigned int i = 0; i < IN; i++) {
            for (unsigned int j = 0; j < OUT; j++) {
                outputLayer[j] += inputLayer[i] * weights[i * OUT + j];
            }
        }
    }

    inline void forward(const std::vector<unsigned int> &featureIndexes, float *outputLayer) {
        memcpy(outputLayer, biases.get(), OUT * sizeof(float));

        for (unsigned int i : featureIndexes) {
            for (unsigned int j = 0; j < OUT; j++) {
                outputLayer[j] += weights[i * OUT + j];
            }
        }
    }
};
