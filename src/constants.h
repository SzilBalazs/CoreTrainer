#ifndef CORETRAINER_SRC_CONSTANTS_H_
#define CORETRAINER_SRC_CONSTANTS_H_

#include <algorithm>
#include <complex>

constexpr float LR = 0.005;
constexpr unsigned int BATCH_SIZE = 8192;
constexpr float EVAL_INFLUENCE = 0.5;
constexpr unsigned int EPOCHS = 500;

constexpr int EVAL_SCALE = 400;
constexpr float EVAL_SCALE_INVERSE = 1.0 / EVAL_SCALE;

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

constexpr float clippedReLU(float in) {
    return std::clamp(in, float(0), float(1));
}

constexpr float clippedReLUDerivative(float in) {
    return 0 < in && in < 1.0;
}

constexpr float sigmoid(float in) {
    return 1.0 / (1.0 + std::exp(-in * EVAL_SCALE_INVERSE));
}

constexpr float sigmoidDerivative(float in) {
    return in * (1 - in) * EVAL_SCALE_INVERSE;
}

constexpr float error(float output, float wdl, float eval) {
    return EVAL_INFLUENCE * (output - eval) * (output - eval) +
           (1 - EVAL_INFLUENCE) * (output - wdl) * (output - wdl);
}

constexpr float errorDerivative(float output, float wdl, float eval) {
    return EVAL_INFLUENCE * 2 * (output - eval) +
           (1 - EVAL_INFLUENCE) * 2 * (output - wdl);
}

#endif //CORETRAINER_SRC_CONSTANTS_H_
