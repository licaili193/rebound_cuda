#include "test_framework.h"
#include <iostream>
#include <string>

// Include all test files
#include "test_physics.cpp"
#include "test_collisions.cpp" 
#include "test_streaming.cpp"

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [category]" << std::endl;
    std::cout << "Categories:" << std::endl;
    std::cout << "  Physics    - Basic physics and gravity tests" << std::endl;
    std::cout << "  Collisions - Collision detection and resolution tests" << std::endl;
    std::cout << "  Streaming  - Data streaming and Observer Pattern tests" << std::endl;
    std::cout << "  (no args)  - Run all tests" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string category = "";
    
    if (argc > 1) {
        category = argv[1];
        
        // Convert to proper case
        if (category == "physics" || category == "Physics" || category == "PHYSICS") {
            category = "Physics";
        } else if (category == "collisions" || category == "Collisions" || category == "COLLISIONS") {
            category = "Collisions";
        } else if (category == "streaming" || category == "Streaming" || category == "STREAMING") {
            category = "Streaming";
        } else if (category == "help" || category == "--help" || category == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cout << "Unknown category: " << category << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Run tests
    if (category.empty()) {
        std::cout << "Running all REBOUND CUDA tests..." << std::endl;
    } else {
        std::cout << "Running " << category << " tests..." << std::endl;
    }
    
    TestFramework::runTests(category);
    
    return TestFramework::getFailedCount();
} 