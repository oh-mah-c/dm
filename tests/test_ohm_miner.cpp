#include "algorithms/ohm_miner.h"
extern "C" {
#include "core/dm_dataset.h"
#include "core/dm_benchmark.h"
#include "algorithms/sparc_hoi.h"
}
#include <iostream>
#include <chrono>

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << " HOI Real Dataset Benchmark: chess.txt\n";
    std::cout << "========================================================\n\n";

    const char* path = "../datasets/itemsets/chess.txt";
    DM_Dataset* ds = dm_dataset_load(path, DM_TYPE_TRANSACTIONAL);
    if (!ds) {
        std::cerr << "Failed to load " << path << "\n";
        return 1;
    }
    std::cout << "[Dataset] Loaded: " << ds->count << " transactions. Max Item ID: " << ds->max_id << "\n";

    // Set a very low min_occupancy to stress-test the algorithms!
    // Dense datasets like chess will cause explosive combinations.
    DM_SPARC_HOI_Params params;
    params.min_occupancy = 0.45; // 45% Occupancy (HOI)
    params.summed_occupancy_mode = 0; // Average Mode
    params.min_support = 0;
    params.max_patterns = 0;
    params.max_seconds = 0.0; // NO TIMEOUT
    
    std::cout << "\n>>> Running SPARC-HOI on chess.txt...\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    sparc_hoi_algo.run(ds, &params);
    auto t2 = std::chrono::high_resolution_clock::now();
    double s1 = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "SPARC-HOI Wall Time: " << s1 << " s\n";
    
    std::cout << "\n>>> Running Ohm-Miner on chess.txt...\n";
    auto t3 = std::chrono::high_resolution_clock::now();
    ohm_miner_algo.run(ds, &params);
    auto t4 = std::chrono::high_resolution_clock::now();
    double s2 = std::chrono::duration<double>(t4 - t3).count();
    std::cout << "Ohm-Miner Wall Time: " << s2 << " s\n";
    
    dm_dataset_free(ds);
    
    return 0;
}
