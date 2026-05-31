#include "algorithms/ohm_bifrost_kernel.h"
#include <iostream>
#include <vector>
#include <fstream>

bool read_bin(const char* path, std::vector<float>& data) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    return true;
}

bool write_bin(const char* path, const std::vector<float>& data) {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.bin>\n";
        return 1;
    }

    std::vector<float> img;
    if (!read_bin(argv[1], img)) {
        std::cerr << "Failed to read " << argv[1] << "\n";
        return 1;
    }
    
    // Check even dimension
    if (img.size() % 2 != 0) {
        img.push_back(0.0f);
    }
    
    std::cout << "[C++] Loaded image array of size: " << img.size() << " floats\n";
    
    // Forward (Scramble)
    dm::algorithm::bifrost_coupling_step_cpu(img.data(), img.size(), true);
    write_bin("scrambled.bin", img);
    std::cout << "[C++] Forward Transform complete -> scrambled.bin\n";
    
    // Reverse (Recover)
    dm::algorithm::bifrost_coupling_step_cpu(img.data(), img.size(), false);
    write_bin("recovered.bin", img);
    std::cout << "[C++] Reverse Transform complete -> recovered.bin\n";

    return 0;
}
