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
constexpr float QUANT_SCALE = 255;

// king buckets * colors * piece types * squares
constexpr unsigned int L_0_SIZE = 4 * 2 * 6 * 64;
constexpr unsigned int L_1_SIZE = 256;
constexpr unsigned int L_2_SIZE = 1;

// clang-format off
constexpr int KING_BUCKET[64]{
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1, 1,
        2, 2, 2, 2, 3, 3, 3, 3,
        2, 2, 2, 2, 3, 3, 3, 3,
        2, 2, 2, 2, 3, 3, 3, 3,
        2, 2, 2, 2, 3, 3, 3, 3,

};
// clang-format on

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

constexpr float sigmoid(float in) {
    return 1.0f / (1.0f + std::exp(-in));
}

constexpr float sigmoidDerivative(float in) {
    float x = sigmoid(in);
    return x * (1 - x);
}

constexpr float error(float output, float wdl, float eval) {
    return EVAL_INFLUENCE * (output - eval) * (output - eval) +
           (1 - EVAL_INFLUENCE) * (output - wdl) * (output - wdl);
}

constexpr float errorDerivative(float output, float wdl, float eval) {
    return EVAL_INFLUENCE * 2 * (output - eval) + (1 - EVAL_INFLUENCE) * 2 * (output - wdl);
}

#endif //CORETRAINER_SRC_CONSTANTS_H_
