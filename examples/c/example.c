/**
 * examples/c/example.c — C usage example for libdm
 *
 * Demonstrates:
 *   - dm_init / dm_version
 *   - Dataset open
 *   - Algorithm run (FP-Growth frequent itemset mining)
 *   - Tokenizer train + encode/decode
 *   - Benchmark API
 *   - BitSet operations
 *   - DataGen (MEDM synthetic)
 *   - CLI passthrough
 *
 * Compile:
 *   gcc -std=c11 -I../../include example.c -L../../ -ldm -o example
 * Run:
 *   LD_LIBRARY_PATH=../../ ./example
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "dm.h"

/* ── helpers ─────────────────────────────────────────────────────────────── */

static void die(const char *msg, DM_Status s) {
    fprintf(stderr, "ERROR %s: %s\n", msg, dm_strerror(s));
    exit(1);
}

#define CHECK(expr, ctx) do { DM_Status _s = (expr); if (_s != DM_OK) die(ctx, _s); } while(0)

/* ── main ────────────────────────────────────────────────────────────────── */

int main(void) {
    /* ── § 1  Core ─────────────────────────────────────────────────── */
    CHECK(dm_init(), "dm_init");
    printf("libdm %s\n", dm_version());

    /* ── § 9  Benchmark start ──────────────────────────────────────── */
    dm_bench_reset();
    dm_bench_start(DM_BENCH_PHASE_TOTAL);

    /* ── § 3  Algorithm: FP-Growth ─────────────────────────────────── */
    {
        DM_Algorithm algo = dm_algorithm_create("fpgrowth");
        if (!algo) { fprintf(stderr, "fpgrowth not available\n"); goto skip_algo; }

        const char *dataset = "../../datasets/itemsets/mushrooms.txt";
        const char *output  = "/tmp/fpgrowth_out.txt";

        DM_Status s = dm_algorithm_run(algo, dataset, output, 0.05, NULL);
        if (s == DM_OK)
            printf("[algorithm] FP-Growth done → %s\n", output);
        else
            fprintf(stderr, "[algorithm] FP-Growth: %s\n", dm_strerror(s));

        dm_algorithm_free(algo);
    }
skip_algo:

    /* ── § 4  Tokenizer: BPE ───────────────────────────────────────── */
    {
        DM_Tokenizer tok = dm_tokenizer_create("bpe");
        if (!tok) { fprintf(stderr, "bpe tokenizer unavailable\n"); goto skip_tok; }

        /* Train */
        const char *corpus = "../../datasets/tokenizer/real_corpus.txt";
        DM_Status s = dm_tokenizer_train(tok, corpus, 1000, "/tmp/bpe_model");
        if (s != DM_OK) {
            fprintf(stderr, "[tokenizer] train: %s\n", dm_strerror(s));
        } else {
            printf("[tokenizer] BPE trained, vocab_size=%d\n",
                   dm_tokenizer_vocab_size(tok));

            /* Encode */
            uint32_t ids[512];
            int len = 512;
            const char *text = "data mining with frequent itemset mining";
            s = dm_tokenizer_encode(tok, text, ids, &len);
            if (s == DM_OK) {
                printf("[tokenizer] '%s' → %d tokens\n", text, len);

                /* Decode */
                char out[1024];
                s = dm_tokenizer_decode(tok, ids, len, out, sizeof(out));
                if (s == DM_OK) printf("[tokenizer] decoded: '%s'\n", out);
            }
        }
        dm_tokenizer_free(tok);
    }
skip_tok:

    /* ── § 10  BitSet ──────────────────────────────────────────────── */
    {
        DM_BitSet a = dm_bitset_create(64);
        DM_BitSet b = dm_bitset_create(64);

        dm_bitset_set(a, 3);
        dm_bitset_set(a, 7);
        dm_bitset_set(b, 7);
        dm_bitset_set(b, 15);
        dm_bitset_and(a, b);   /* a ∩ b = {7} */

        printf("[bitset] popcount after AND = %zu (expected 1)\n",
               dm_bitset_popcount(a));

        dm_bitset_free(a);
        dm_bitset_free(b);
    }

    /* ── § 11  DataGen: MEDM ───────────────────────────────────────── */
    {
        DM_DataGen gen = dm_datagen_create("medm");
        if (gen) {
            DM_Status s = dm_datagen_run(gen, "nItems=100,nTxn=500,avgLen=10",
                                         "/tmp/syn_medm.txt", 42);
            if (s == DM_OK)
                printf("[datagen] MEDM synthetic → /tmp/syn_medm.txt\n");
            else
                fprintf(stderr, "[datagen] MEDM: %s\n", dm_strerror(s));
            dm_datagen_free(gen);
        }
    }

    /* ── § 9  Benchmark stop + report ─────────────────────────────── */
    dm_bench_stop(DM_BENCH_PHASE_TOTAL);
    dm_bench_print("c_example", "mixed");

    /* ── § 5  Vision ────────────────────────────────────────────────── */
    {
        DM_Vision vis = dm_vision_create("mobilenet_tiny");
        if (vis) {
            DM_Status s = dm_vision_init(vis, "/tmp/mobilenet_ckpt", 1000, 224, 1.0f, 0.001f);
            if (s != DM_OK) {
                printf("[vision] skipped — init failed: %s\n", dm_strerror(s));
            } else {
                float *dummy = calloc(224 * 224 * 3, sizeof(float));
                float probs[1000] = {0};
                s = dm_vision_predict(vis, dummy, 224, 224, probs, 1000);
                if (s == DM_OK) {
                    printf("[vision] predicted probs length: 1000\n");
                }
                free(dummy);
            }
            dm_vision_free(vis);
        }
    }

    /* ── § 6  Language Model ────────────────────────────────────────── */
    {
        DM_LM lm = dm_lm_create("tinystories");
        if (lm) {
            DM_Status s = dm_lm_load(lm, "/tmp/tinystories_ckpt");
            if (s != DM_OK) {
                printf("[lm] skipped — no checkpoint: %s\n", dm_strerror(s));
            } else {
                char buf[512] = {0};
                s = dm_lm_generate(lm, "Once upon a time", 64, buf, sizeof(buf));
                if (s == DM_OK) {
                    printf("[lm] generated: %s\n", buf);
                }
            }
            dm_lm_free(lm);
        }
    }

    /* ── § 13  CLI passthrough ─────────────────────────────────────── */
    printf("\n--- dm CLI: algorithm list (first 3 lines) ---\n");
    char buf[16384];
    dm_algorithm_list(buf, sizeof(buf));
    /* Print only first 3 IDs */
    char *p = buf;
    for (int i = 0; i < 3 && *p; i++) {
        char *nl = strchr(p, '\n');
        if (nl) *nl = '\0';
        printf("  %s\n", p);
        if (nl) p = nl + 1; else break;
    }

    return 0;
}
