#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "MLAS/include/mlas.h"

namespace {

void FillRandom(std::vector<float>& data, float scale = 1.0f) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-scale, scale);
    for (float& v : data) {
        v = dist(rng);
    }
}

double BenchmarkGemm(size_t M, size_t N, size_t K, int iterations) {
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C(M * N);
    FillRandom(A);
    FillRandom(B);

    // Warm-up
    MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K,
             1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N, nullptr);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        MlasGemm(CblasNoTrans, CblasNoTrans, M, N, K,
                 1.0f, A.data(), K, B.data(), N, 0.0f, C.data(), N, nullptr);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

double BenchmarkFlashAttention(int batch_size,
                               int num_heads,
                               int q_sequence_length,
                               int kv_sequence_length,
                               int qk_head_size,
                               int v_head_size,
                               int q_block_size,
                               int kv_block_size,
                               int iterations) {
    const int thread_count = 1;  // simple single-thread run

    size_t q_count = static_cast<size_t>(batch_size) * num_heads * q_sequence_length * qk_head_size;
    size_t kv_count = static_cast<size_t>(batch_size) * num_heads * kv_sequence_length;

    std::vector<float> query(q_count);
    std::vector<float> key(kv_count * qk_head_size);
    std::vector<float> value(kv_count * v_head_size);
    std::vector<float> output(static_cast<size_t>(batch_size) * q_sequence_length * num_heads * v_head_size);

    FillRandom(query, 0.1f);
    FillRandom(key, 0.1f);
    FillRandom(value, 0.1f);

    size_t floats_per_thread = static_cast<size_t>(q_block_size) *
        (2 + kv_block_size + v_head_size);
    std::vector<float> buffer(floats_per_thread * thread_count, 0.0f);

    MlasFlashAttentionThreadedArgs args{};
    args.batch_size = batch_size;
    args.num_heads = num_heads;
    args.q_sequence_length = q_sequence_length;
    args.kv_sequence_length = kv_sequence_length;
    args.qk_head_size = qk_head_size;
    args.v_head_size = v_head_size;
    args.q_block_size = q_block_size;
    args.kv_block_size = kv_block_size;
    args.scale = 1.0f / std::sqrt(static_cast<float>(qk_head_size));
    args.thread_count = thread_count;
    args.buffer = buffer.data();
    args.buffer_size_per_thread = floats_per_thread * sizeof(float);
    args.query = query.data();
    args.key = key.data();
    args.value = value.data();
    args.output = output.data();

    // Warm-up
    MlasFlashAttention(&args, nullptr);

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        MlasFlashAttention(&args, nullptr);
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count() / iterations;
}

}  // namespace

int main() {
    const int iterations = 10;

    size_t M = 512, N = 512, K = 512;
    double gemm_ms = BenchmarkGemm(M, N, K, iterations);
    std::cout << "GEMM " << M << "x" << N << "x" << K << " avg ms: " << gemm_ms << "\n";

    int batch = 1;
    int heads = 4;
    int q_seq = 128;
    int kv_seq = 128;
    int qk_head = 64;
    int v_head = 64;
    int q_block = 64;
    int kv_block = 64;
    double flash_ms = BenchmarkFlashAttention(batch, heads, q_seq, kv_seq,
                                              qk_head, v_head, q_block, kv_block,
                                              iterations);
    std::cout << "FlashAttention batch=" << batch
              << " heads=" << heads
              << " q_seq=" << q_seq
              << " kv_seq=" << kv_seq
              << " head_dim=" << qk_head
              << " avg ms: " << flash_ms << "\n";

    return 0;
}
