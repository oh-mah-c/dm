/**
 * examples/cpp/example.cpp — C++ usage example for libdm
 *
 * Demonstrates RAII wrappers from bindings/cpp/dm.hpp:
 *   - dm::Algorithm  (FP-Growth, EFIM)
 *   - dm::Tokenizer  (BPE encode/decode)
 *   - dm::Tensor     (alloc, fill, ops)
 *   - dm::BitSet
 *   - dm::DataGen
 *   - dm::GpuCtx     (device name)
 *   - dm::Arena
 *
 * Compile:
 *   g++ -std=c++17 -I../../include -I../../bindings/cpp \
 *       example.cpp -L../../ -ldm -o example_cpp
 * Run:
 *   LD_LIBRARY_PATH=../../ ./example_cpp
 */

#include "dm.hpp"    /* pulls in dm.h automatically */
#include <iostream>
#include <vector>

int main() {
    try {
        dm::init();
        std::cout << "libdm " << dm::version() << "\n\n";

        // ── Algorithm: FP-Growth ────────────────────────────────────────────
        {
            dm::Algorithm algo("fpgrowth");
            algo.run("../../datasets/itemsets/mushrooms.txt",
                     "/tmp/fpgrowth_cpp_out.txt",
                     0.05,
                     {"min_length=2", "max_length=10"});
            std::cout << "[algorithm] FP-Growth done\n";
        }

        // ── Algorithm: EFIM (high-utility) ─────────────────────────────────
        {
            dm::Algorithm efim("efim");
            efim.run("../../datasets/utilities/foodmart.txt",
                     "/tmp/efim_out.txt",
                     /*min_utility=*/50.0);
            std::cout << "[algorithm] EFIM done\n";
        }

        // ── List all algorithms ─────────────────────────────────────────────
        {
            auto list = dm::Algorithm::list();
            // Print first 5 IDs
            std::string id;
            int cnt = 0;
            for (char c : list) {
                if (c == '\n') { std::cout << "  " << id << "\n"; id.clear(); if (++cnt == 5) break; }
                else id += c;
            }
        }

        // ── Tokenizer ───────────────────────────────────────────────────────
        {
            dm::Tokenizer tok("bpe");
            tok.train("../../datasets/tokenizer/real_corpus.txt",
                      2000, "/tmp/bpe_cpp_model");
            std::cout << "[tokenizer] vocab_size=" << tok.vocab_size() << "\n";

            auto ids = tok.encode("frequent itemset mining is powerful");
            std::cout << "[tokenizer] encoded " << ids.size() << " tokens\n";

            auto text = tok.decode(ids);
            std::cout << "[tokenizer] decoded: " << text << "\n";
        }

        // ── Tensor ops ──────────────────────────────────────────────────────
        {
            dm::Tensor t(1, 3, 4, 4);   // NCHW
            t.fill(1.0f);
            std::cout << "[tensor] count=" << t.count()
                      << " val[0,0,0,0]=" << t.get(0, 0, 0, 0) << "\n";
            dm_op_relu6(&t.raw());
            std::cout << "[tensor] relu6 applied\n";
        }

        // ── BitSet ──────────────────────────────────────────────────────────
        {
            dm::BitSet a(128), b(128);
            a.set(10); a.set(42); a.set(99);
            b.set(42); b.set(99); b.set(127);
            a &= b;
            std::cout << "[bitset] popcount=" << a.popcount()
                      << " (expected 2: {42,99})\n";
        }

        // ── DataGen ─────────────────────────────────────────────────────────
        {
            dm::DataGen gen("medm");
            gen.run("nItems=50,nTxn=200,avgLen=8",
                    "/tmp/syn_cpp.txt", 0x1234);
            std::cout << "[datagen] MEDM synthetic written\n";
        }

        // ── GPU context ─────────────────────────────────────────────────────
        {
            dm::GpuCtx gpu;
            std::cout << "[gpu] ready=" << gpu.ready()
                      << " device='" << gpu.device_name() << "'\n";
        }

        // ── Arena ───────────────────────────────────────────────────────────
        {
            dm::Arena arena(1024 * 1024);   // 1 MB
            auto *buf = static_cast<float *>(arena.alloc(64 * sizeof(float), 4));
            std::fill(buf, buf + 64, 3.14f);
            std::cout << "[arena] allocated 64 floats, first=" << buf[0] << "\n";
        }

        // ── Benchmark ───────────────────────────────────────────────────────
        dm::bench_reset();
        dm::bench_start(DM_BENCH_PHASE_TOTAL);
        {
            dm::Algorithm algo("apriori");
            algo.run("../../datasets/itemsets/T10I4D100K.txt",
                     "/tmp/apriori_cpp.txt", 0.1);
        }
        dm::bench_stop(DM_BENCH_PHASE_TOTAL);
        dm::bench_print("apriori", "T10I4D100K");

        auto r = dm::bench_report();
        std::cout << "[bench] total=" << r.phase_ms[3] << "ms "
                  << "patterns=" << r.num_patterns << "\n";

    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
