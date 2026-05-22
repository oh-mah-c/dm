/*
 * unigram_word.glsl — Unigram EM forward-backward per word
 *
 * One workgroup = one word (local_size_x = 1, one thread does the whole DP).
 * This keeps the code simple and gives parallelism across words.
 * For long words (> MAX_WORD_LEN) the kernel skips gracefully.
 *
 * Requires: VK_EXT_shader_atomic_float (for atomicAdd on float in new_counts).
 *
 * Compile:
 *   glslc --target-env=vulkan1.1 -O unigram_word.glsl -o unigram_word.spv
 */

#version 450
#extension GL_EXT_shader_atomic_float : require

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PC {
    uint n_words;
    uint n_pieces;
    uint max_word_len;
    uint _pad;
} pc;

/* cp_ids packed: two uint16 values per uint32, little-endian */
layout(set = 0, binding = 0) readonly buffer LogProbs    { float  log_probs[];    };
layout(set = 0, binding = 1) readonly buffer CpIds       { uint   cp_ids_packed[];};
layout(set = 0, binding = 2) readonly buffer WordStarts  { uint   word_starts[];  };
layout(set = 0, binding = 3) readonly buffer WordLens    { uint   word_lens[];    };
layout(set = 0, binding = 4) readonly buffer WordFreqs   { uint   word_freqs[];   };
layout(set = 0, binding = 5) readonly buffer PieceRowPtr { uint   piece_row_ptr[];};
layout(set = 0, binding = 6) readonly buffer PieceCol    { uint   piece_col[];    };
layout(set = 0, binding = 7)          buffer NewCounts   { float  new_counts[];   };
layout(set = 0, binding = 8)          buffer ExpTotal    { float  exp_total[];    };

const float LOG_ZERO = -1e30;
const uint  MAX_LEN  = 512u;

/* Local dynamic programming tables — max word length limited to MAX_LEN */
float fwd[MAX_LEN + 1u];
float bwd[MAX_LEN + 1u];

/* Safe log-sum-exp of two values */
float logsum(float a, float b) {
    if (a == LOG_ZERO) return b;
    if (b == LOG_ZERO) return a;
    if (a > b) return a + log(1.0 + exp(b - a));
    return b + log(1.0 + exp(a - b));
}

/* Unpack uint16 from packed uint32 buffer */
uint get_cp(uint packed_base, uint cp_idx) {
    uint word32 = cp_ids_packed[(packed_base + cp_idx) / 2u];
    uint shift  = ((cp_idx & 1u) == 0u) ? 0u : 16u;
    return (word32 >> shift) & 0xFFFFu;
}

void main() {
    uint w = gl_GlobalInvocationID.x;
    if (w >= pc.n_words) return;

    uint cp_start = word_starts[w];   /* start index in cp_ids */
    uint cp_len   = word_lens[w];     /* codepoint length of word */
    uint freq     = word_freqs[w];

    /* Skip words longer than our stack allocation */
    if (cp_len > MAX_LEN) return;

    /* Forward pass: fwd[i] = log P(prefix of length i) */
    for (uint i = 0u; i <= cp_len; i++) fwd[i] = LOG_ZERO;
    fwd[0] = 0.0;

    for (uint pos = 0u; pos < cp_len; pos++) {
        if (fwd[pos] == LOG_ZERO) continue;
        /* Iterate over pieces that start at this position */
        uint arc_start = piece_row_ptr[cp_start + pos];
        uint arc_end   = piece_row_ptr[cp_start + pos + 1u];
        for (uint k = arc_start; k < arc_end; k++) {
            uint  piece_id  = piece_col[k];
            /* Piece length is encoded as (piece_id >> 24)+1 — see CPU side */
            /* For simplicity, the CPU builds the coverage table with arc lengths */
            /* embedded; here we just trust arc_end - arc_start ordering.        */
            /* Actually piece lengths come from the model's max_piece_len.        */
            /* We scan forward from pos to find the end position.                */
            /* The CPU must encode (piece_id, end_pos) pairs in piece_col[].     */
            /* Convention used here: piece_col stores interleaved (piece_id, end_pos) */
            /* so arc_end - arc_start is halved and k steps by 2. */
            uint end_pos = piece_col[k + 1u];  /* end codepoint position */
            k++;                                /* consume the end_pos entry */
            if (end_pos > cp_len) continue;
            float lp = log_probs[piece_id];
            fwd[end_pos] = logsum(fwd[end_pos], fwd[pos] + lp);
        }
    }

    float z = fwd[cp_len];
    if (z == LOG_ZERO) return; /* word has no valid segmentation */

    /* Backward pass: bwd[i] = log P(suffix starting at i) */
    for (uint i = 0u; i <= cp_len; i++) bwd[i] = LOG_ZERO;
    bwd[cp_len] = 0.0;

    for (uint pos = cp_len; pos > 0u; ) {
        pos--;
        if (bwd[pos + 1u] == LOG_ZERO && pos + 1u != cp_len) continue;
        /* Scan backwards: find arcs ending at pos+1 */
        /* (same coverage table, just traverse in reverse) */
        uint arc_start = piece_row_ptr[cp_start + pos];
        uint arc_end   = piece_row_ptr[cp_start + pos + 1u];
        for (uint k = arc_start; k < arc_end; k += 2u) {
            uint piece_id = piece_col[k];
            uint end_pos  = piece_col[k + 1u];
            if (end_pos > cp_len) continue;
            float lp = log_probs[piece_id];
            bwd[pos] = logsum(bwd[pos], lp + bwd[end_pos]);
        }
    }

    /* Scatter expected counts: for each arc, posterior = fwd[s] + lp + bwd[e] - z */
    float freq_f = float(freq);
    float total_contrib = 0.0;

    for (uint pos = 0u; pos < cp_len; pos++) {
        if (fwd[pos] == LOG_ZERO) continue;
        uint arc_start = piece_row_ptr[cp_start + pos];
        uint arc_end   = piece_row_ptr[cp_start + pos + 1u];
        for (uint k = arc_start; k < arc_end; k += 2u) {
            uint  piece_id = piece_col[k];
            uint  end_pos  = piece_col[k + 1u];
            if (end_pos > cp_len) continue;
            float lp       = log_probs[piece_id];
            float posterior = exp(fwd[pos] + lp + bwd[end_pos] - z);
            float contrib   = posterior * freq_f;
            atomicAdd(new_counts[piece_id], contrib);
            total_contrib += contrib;
        }
    }

    atomicAdd(exp_total[0], total_contrib);
}
