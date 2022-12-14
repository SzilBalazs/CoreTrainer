#ifndef CORETRAINER_SRC_CONSTANTS_H_
#define CORETRAINER_SRC_CONSTANTS_H_

#include <algorithm>
#include <complex>

extern float LR;
constexpr unsigned int BATCH_SIZE = 16384;
constexpr float EVAL_INFLUENCE = 0.9;
constexpr unsigned int EPOCHS = 500;

constexpr float BETA1 = 0.9;
constexpr float BETA2 = 0.999;
constexpr float EPSILON = 1e-8;

constexpr int ITERATIONS_PER_CHECKPOINT = 200;

constexpr int THREAD_COUNT = 4;

constexpr float EVAL_SCALE = 400;
constexpr float EVAL_SCALE_INVERSE = 1 / EVAL_SCALE;
constexpr float QUANT_SCALE = 64;

constexpr unsigned int L_0_SIZE = 2 * 6 * 64;
constexpr unsigned int L_1_SIZE = 256;
constexpr unsigned int L_2_SIZE = 1;

enum Color {
    WHITE = 1,
    BLACK = 0
};

enum PieceType {
    KING = 0,
    PAWN = 1,
    KNIGHT = 2,
    BISHOP = 3,
    ROOK = 4,
    QUEEN = 5
};

constexpr float ReLU(float in) {
    return std::max(in, 0.0f);
}

constexpr float ReLUDerivative(float in) {
    return 0 < in;
}

constexpr float clippedReLU(float in) {
    return std::clamp(in, 0.0f, 64.0f);
}

constexpr float clippedReLUDerivative(float in) {
    return 0.0f < in && in < 64.0f;
}

constexpr float sigmoid(float in) {
    return 1.0f / (1.0f + std::exp(-in * EVAL_SCALE_INVERSE));
}

constexpr float sigmoidDerivative(float in) {
    float x = sigmoid(in);
    return x * (1 - x) * EVAL_SCALE_INVERSE;
}

constexpr float error(float output, float expected) {
    return (output - expected) * (output - expected);
}

constexpr float errorDerivative(float output, float expected) {
    return 2 * (output - expected);
}

#endif //CORETRAINER_SRC_CONSTANTS_H_
