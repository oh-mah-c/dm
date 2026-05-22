/*
 * bpe_pair_count.glsl — BPE pair-frequency histogram
 *
 * Each invocation handles one word.
 * Atomically increments pair_counts[left * vocab_size + right] for every
 * adjacent pair (left, right) in the word, weighted by the word's frequency.
 *
 * Compile: glslc --target-env=vulkan1.1 -O bpe_pair_count.glsl -o bpe_pair_count.spv
 */

#version 450
#extension GL_KHR_memory_scope_semantics : enable

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

/* Push constants */
layout(push_constant) uniform PC {
    uint n_words;
    uint vocab_size;
} pc;

/* Storage buffers */
layout(set = 0, binding = 0) readonly buffer SymIds    { uint sym_ids[];    };
layout(set = 0, binding = 1) readonly buffer WordStart { uint word_starts[];};
layout(set = 0, binding = 2) readonly buffer WordLen   { uint word_lens[];  };
layout(set = 0, binding = 3) readonly buffer WordFreq  { uint word_freqs[]; };
layout(set = 0, binding = 4)          buffer PairCount { uint pair_counts[];};

void main() {
    uint w = gl_GlobalInvocationID.x;
    if (w >= pc.n_words) return;

    uint off  = word_starts[w];
    uint len  = word_lens[w];
    uint freq = word_freqs[w];

    for (uint i = 0u; i + 1u < len; i++) {
        uint a = sym_ids[off + i];
        uint b = sym_ids[off + i + 1u];
        atomicAdd(pair_counts[a * pc.vocab_size + b], freq);
    }
}
