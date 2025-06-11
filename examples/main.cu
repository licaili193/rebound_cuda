#include "../include/rebound_cuda.h"
#include <iostream>

// Example function declarations
extern void runPlanetarySystemExample();
extern void runGravityModeComparison();

int main() {
    // Print device information
    printDeviceInfo();
    
    try {
        // Run planetary system example
        runPlanetarySystemExample();
        
        // Run gravity mode comparison
        runGravityModeComparison();
        
        std::cout << "\n=== ALL EXAMPLES COMPLETED SUCCESSFULLY! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error running examples: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 