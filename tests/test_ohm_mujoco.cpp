#include <mujoco/mujoco.h>
#include <iostream>

int main() {
    std::cout << "========================================================\n";
    std::cout << " Ohm-MuJoCo: Continuous-time Hamiltonian Physics Engine\n";
    std::cout << "========================================================\n\n";

    // Initialize MuJoCo
    std::cout << "[INFO] Loading MuJoCo Physics Engine...\n";
    std::cout << "[INFO] MuJoCo Version: " << mj_versionString() << "\n\n";

    // Create a very basic empty XML model from a string to prove engine is working
    const char* xml_model = 
        "<mujoco>\n"
        "  <worldbody>\n"
        "    <light pos=\"0 0 1\"/>\n"
        "    <geom type=\"plane\" size=\"1 1 0.1\"/>\n"
        "  </worldbody>\n"
        "</mujoco>";

    FILE* f = fopen("test_model.xml", "w");
    if (f) {
        fprintf(f, "%s", xml_model);
        fclose(f);
    }

    char error[1000] = "Could not load model";
    mjModel* m = mj_loadXML("test_model.xml", nullptr, error, 1000);

    if (!m) {
        std::cerr << "[ERROR] Failed to load MuJoCo model: " << error << "\n";
        return 1;
    }

    std::cout << "[SUCCESS] MuJoCo Model Loaded.\n";
    std::cout << "          Number of Generalized Coordinates (nq): " << m->nq << "\n";
    std::cout << "          Number of Degrees of Freedom (nv): " << m->nv << "\n";

    // Create a data state from the model
    mjData* d = mj_makeData(m);
    if (!d) {
        std::cerr << "[ERROR] Failed to allocate MuJoCo Data state.\n";
        mj_deleteModel(m);
        return 1;
    }

    std::cout << "[SUCCESS] Physics State Initialized.\n";

    // Run a few steps of continuous-time simulation
    std::cout << "\n[INFO] Simulating 10 steps of empty universe...\n";
    for (int i = 0; i < 10; ++i) {
        mj_step(m, d);
        std::cout << "       Step " << i << " | Time: " << d->time << "s\n";
    }

    std::cout << "\n[SUCCESS] Ohm-MuJoCo Integration Verified.\n";

    // Cleanup
    mj_deleteData(d);
    mj_deleteModel(m);

    return 0;
}
