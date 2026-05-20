#ifndef DM_LAGA_H
#define DM_LAGA_H

#include "core/dm_benchmark.h"
#include "core/dm_arena.h"

int dm_laga_cli(int argc, char **argv);

/**
 * @brief Generate synthetic dataset directly into RAM (Arena Allocator)
 * @param arena The arena allocator to use
 * @param target_bytes The approximate target size of the dataset in bytes
 * @param alpha Zipfian distribution skew parameter (e.g. 0.5)
 * @param noise Noise parameter for utility generation (if any)
 * @return BenchmarkDataset* allocated in the arena
 */
BenchmarkDataset* dm_laga_generate_ram(DM_Arena* arena, size_t target_bytes, double alpha, double noise);

#endif
