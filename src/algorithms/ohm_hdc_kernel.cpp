#include "algorithms/ohm_hdc_kernel.h"
#include <cstring>
#include <stdexcept>

namespace dm {
namespace ai {

void OhmHDC::generate_random_hv(HyperVector& out, std::mt19937& rng) {
    // Generate random 32-bit integers to fill the 256-bit blocks
    std::uniform_int_distribution<uint32_t> dist(0, 0xFFFFFFFF);
    
    uint32_t* raw_ptr = reinterpret_cast<uint32_t*>(&out);
    for (int i = 0; i < HV_BLOCKS * 8; ++i) { // 8 x 32-bit = 256-bit
        raw_ptr[i] = dist(rng);
    }
}

void OhmHDC::bind(const HyperVector& A, const HyperVector& B, HyperVector& out) {
    for (int i = 0; i < HV_BLOCKS; ++i) {
        // AVX2 XOR operation: 256 bits per cycle!
        out.blocks[i] = _mm256_xor_si256(A.blocks[i], B.blocks[i]);
    }
}

void OhmHDC::bundle(const std::vector<HyperVector>& vecs, HyperVector& out) {
    if (vecs.empty()) return;
    
    // We need to count 1s across all vectors for each bit position.
    // To do this simply in C++ without crazy bit slicing:
    int counts[HV_BITS] = {0};
    
    for (const auto& hv : vecs) {
        const uint64_t* ptr = reinterpret_cast<const uint64_t*>(&hv);
        for (int i = 0; i < HV_BITS / 64; ++i) {
            uint64_t val = ptr[i];
            for (int bit = 0; bit < 64; ++bit) {
                if ((val >> bit) & 1) {
                    counts[i * 64 + bit]++;
                } else {
                    counts[i * 64 + bit]--;
                }
            }
        }
    }
    
    // Reconstruct output bit by bit based on majority rule
    uint64_t* out_ptr = reinterpret_cast<uint64_t*>(&out);
    std::memset(out_ptr, 0, HV_BITS / 8);
    
    for (int i = 0; i < HV_BITS / 64; ++i) {
        uint64_t block = 0;
        for (int bit = 0; bit < 64; ++bit) {
            // Majority rule: > 0 means more 1s than 0s
            // Ties (== 0) can be assigned randomly or to 0
            if (counts[i * 64 + bit] > 0) {
                block |= (1ULL << bit);
            }
        }
        out_ptr[i] = block;
    }
}

void OhmHDC::permute(const HyperVector& in, HyperVector& out, int shift_amount) {
    // A cyclic shift of the entire 10240 bit array.
    // For simplicity, we implement a 1-bit cyclic left shift across 64-bit blocks
    const uint64_t* in_ptr = reinterpret_cast<const uint64_t*>(&in);
    uint64_t* out_ptr = reinterpret_cast<uint64_t*>(&out);
    
    int num_words = HV_BITS / 64;
    for (int i = 0; i < num_words; ++i) {
        int next_idx = (i + 1) % num_words;
        // Shift left by 1, bring in MSB of the next block
        out_ptr[i] = (in_ptr[i] << 1) | (in_ptr[next_idx] >> 63);
    }
}

int OhmHDC::hamming_distance(const HyperVector& A, const HyperVector& B) {
    int distance = 0;
    for (int i = 0; i < HV_BLOCKS; ++i) {
        __m256i diff = _mm256_xor_si256(A.blocks[i], B.blocks[i]);
        
        // Extract 4x 64-bit integers and popcnt
        uint64_t v0 = _mm256_extract_epi64(diff, 0);
        uint64_t v1 = _mm256_extract_epi64(diff, 1);
        uint64_t v2 = _mm256_extract_epi64(diff, 2);
        uint64_t v3 = _mm256_extract_epi64(diff, 3);
        
        distance += __builtin_popcountll(v0);
        distance += __builtin_popcountll(v1);
        distance += __builtin_popcountll(v2);
        distance += __builtin_popcountll(v3);
    }
    return distance;
}

double OhmHDC::cosine_similarity(const HyperVector& A, const HyperVector& B) {
    int dist = hamming_distance(A, B);
    // 1.0 = identical (dist == 0)
    // 0.0 = orthogonal (dist == HV_BITS/2)
    // -1.0 = opposite (dist == HV_BITS)
    return 1.0 - (2.0 * dist / (double)HV_BITS);
}

} // namespace ai
} // namespace dm
