#include "algorithms/ohm_hdc_kernel.h"
#include <iostream>
#include <iomanip>

using namespace dm::ai;

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-HDC: Hyperdimensional Computing Engine (10240-bit)\n";
    std::cout << "========================================================\n\n";

    std::mt19937 rng(42); // Deterministic seed for reproducible tests

    // 1. Generate base orthogonal concepts
    HyperVector V_COUNTRY, V_VIETNAM, V_CAPITAL, V_HANOI, V_FRANCE, V_PARIS;
    
    OhmHDC::generate_random_hv(V_COUNTRY, rng);
    OhmHDC::generate_random_hv(V_VIETNAM, rng);
    OhmHDC::generate_random_hv(V_CAPITAL, rng);
    OhmHDC::generate_random_hv(V_HANOI, rng);
    OhmHDC::generate_random_hv(V_FRANCE, rng);
    OhmHDC::generate_random_hv(V_PARIS, rng);

    // Verify orthogonality
    int dist_rand = OhmHDC::hamming_distance(V_COUNTRY, V_VIETNAM);
    std::cout << "[INFO] Distance between unrelated vectors (COUNTRY vs VIETNAM): " 
              << dist_rand << " bits (Expected ~5120)\n\n";

    // 2. Binding (XOR)
    std::cout << "[1] Binding Concepts...\n";
    HyperVector R_VIETNAM_INFO, R_FRANCE_INFO;
    
    // R1 = (COUNTRY * VIETNAM) + (CAPITAL * HANOI)
    HyperVector bind_c_v, bind_cap_h;
    OhmHDC::bind(V_COUNTRY, V_VIETNAM, bind_c_v);
    OhmHDC::bind(V_CAPITAL, V_HANOI, bind_cap_h);
    
    // R2 = (COUNTRY * FRANCE) + (CAPITAL * PARIS)
    HyperVector bind_c_f, bind_cap_p;
    OhmHDC::bind(V_COUNTRY, V_FRANCE, bind_c_f);
    OhmHDC::bind(V_CAPITAL, V_PARIS, bind_cap_p);

    // 3. Bundling (Majority Rule)
    std::cout << "[2] Bundling memories into singular HyperVectors...\n";
    OhmHDC::bundle({bind_c_v, bind_cap_h}, R_VIETNAM_INFO);
    OhmHDC::bundle({bind_c_f, bind_cap_p}, R_FRANCE_INFO);

    // Let's create a global "World Knowledge Base" by bundling the two country records
    // Note: Bundling just 2 vectors can cause ties, usually we bundle 3+. 
    // Here we bundle them to show the superposition property.
    HyperVector WORLD_MEMORY;
    OhmHDC::bundle({R_VIETNAM_INFO, R_FRANCE_INFO, V_COUNTRY}, WORLD_MEMORY); // Add V_COUNTRY as noise breaker

    std::cout << "[3] Memory recorded. Brain state updated.\n\n";

    // 4. Querying the System: "What is the capital of Vietnam?"
    // Math: Query = WORLD_MEMORY (*) V_COUNTRY (*) V_VIETNAM
    // Wait, simpler: We query the specific R_VIETNAM_INFO memory first.
    std::cout << "[4] Querying: 'What is the CAPITAL in the Vietnam record?'\n";
    HyperVector query_result;
    OhmHDC::bind(R_VIETNAM_INFO, V_CAPITAL, query_result);

    // Check distances to all known concepts
    int d_hanoi = OhmHDC::hamming_distance(query_result, V_HANOI);
    int d_paris = OhmHDC::hamming_distance(query_result, V_PARIS);
    int d_vietnam = OhmHDC::hamming_distance(query_result, V_VIETNAM);
    
    std::cout << "   Distance to HANOI   : " << d_hanoi << " bits (Should be lowest)\n";
    std::cout << "   Distance to PARIS   : " << d_paris << " bits\n";
    std::cout << "   Distance to VIETNAM : " << d_vietnam << " bits\n\n";

    if (d_hanoi < 4000) {
        std::cout << "[SUCCESS] The system successfully reasoned that HANOI is the capital!\n";
        return 0;
    } else {
        std::cout << "[FAILED] The reasoning failed.\n";
        return 1;
    }
}
