#include "agent_loop.h"
#include "agent_trainer.h"
#include "../../../core/gpu/dm_hologram.h"
#include <mujoco/mujoco.h>
#include <iostream>
#include <thread>
#include <chrono>

using namespace ohm::nlp;

int main(int argc, char** argv) {
    std::cout << "==========================================\n";
    std::cout << "   DM NATIVE AGENTIC INFERENCE ENGINE\n";
    std::cout << "==========================================\n";

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_gguf_model> [path_to_mujoco_xml]\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string xml_path = (argc > 2) ? argv[2] : "";

    // 1. Initialize Agent (Loads Weights into dm_torch)
    AgentLoop agent(model_path);

    // 2. Initialize Physics Engine (MuJoCo) if XML provided
    mjModel* m = nullptr;
    mjData* d = nullptr;
    
    if (!xml_path.empty()) {
        char error[1000] = "Could not load XML model";
        m = mj_loadXML(xml_path.c_str(), 0, error, 1000);
        if (!m) {
            std::cerr << "[Physics] Load model error: " << error << "\n";
            return 1;
        }
        d = mj_makeData(m);
        std::cout << "[Physics] Loaded world from " << xml_path << "\n";
        
        // Connect the bridge
        agent.attach_physics(m, d);
    } else {
        std::cout << "[Warning] No physics world provided. Agent will run disembodied.\n";
    }

    // 2.5 Initialize Holographic Display
    std::unique_ptr<ohm::gpu::HologramRenderer> renderer;
    if (m && d) {
        renderer = std::make_unique<ohm::gpu::HologramRenderer>(m, d);
    }
    
    // 2.8 Initialize and Start Agent Trainer (Phase 5)
    ohm::nlp::AgentTrainer trainer(agent.get_replay_buffer(),
                                   agent.get_llm(),
                                   agent.get_state_encoder(),
                                   agent.get_action_head());
    trainer.start();

    // 3. Autonomous Loop
    std::cout << ">>> Initiating Autonomous Loop...\n";
    int max_steps = 10000; // Let it run longer for visualization
    for (int i = 0; i < max_steps; ++i) {
        if (renderer && renderer->should_close()) {
            std::cout << "[Hologram] Window closed by user.\n";
            break;
        }

        agent.step();
        
        // If physics is attached, step the simulation based on the agent's actions
        if (m && d && agent.get_state() == AgentState::LEARNING) {
            // Agent just Acted, now Physics steps forward
            mj_step(m, d);
            std::cout << "--- [Time: " << d->time << "s] Physics advanced.\n";
            
            if (renderer) renderer->render();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // Stop trainer gracefully
    trainer.stop();

    std::cout << ">>> Autonomous Loop terminated gracefully.\n";

    if (d) mj_deleteData(d);
    if (m) mj_deleteModel(m);

    return 0;
}
