#ifndef TEST_FRAMEWORK_H
#define TEST_FRAMEWORK_H

#include <iostream>
#include <vector>
#include <string>
#include <functional>
#include <chrono>
#include <cmath>

// Simple test framework for REBOUND CUDA
class TestFramework {
private:
    struct Test {
        std::string name;
        std::function<void()> test_func;
        std::string category;
    };
    
    static std::vector<Test> tests_;
    static int passed_;
    static int failed_;
    static std::string current_test_;
    
public:
    static void addTest(const std::string& name, const std::string& category, std::function<void()> test_func) {
        tests_.push_back({name, test_func, category});
    }
    
    static void runTests(const std::string& category = "") {
        std::cout << "Running REBOUND CUDA Unit Tests" << std::endl;
        std::cout << "===============================" << std::endl;
        
        passed_ = 0;
        failed_ = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (const auto& test : tests_) {
            if (!category.empty() && test.category != category) {
                continue;
            }
            
            current_test_ = test.name;
            std::cout << "[" << test.category << "] Running: " << test.name << "... ";
            
            try {
                test.test_func();
                std::cout << "PASSED" << std::endl;
                passed_++;
            } catch (const std::exception& e) {
                std::cout << "FAILED: " << e.what() << std::endl;
                failed_++;
            } catch (...) {
                std::cout << "FAILED: Unknown exception" << std::endl;
                failed_++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "\n===============================" << std::endl;
        std::cout << "Test Results:" << std::endl;
        std::cout << "  Passed: " << passed_ << std::endl;
        std::cout << "  Failed: " << failed_ << std::endl;
        std::cout << "  Total:  " << (passed_ + failed_) << std::endl;
        std::cout << "  Time:   " << duration.count() << " ms" << std::endl;
        
        if (failed_ == 0) {
            std::cout << "All tests PASSED!" << std::endl;
        } else {
            std::cout << "Some tests FAILED!" << std::endl;
        }
    }
    
    static int getFailedCount() { return failed_; }
};

// Test assertion macros
#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            throw std::runtime_error("Assertion failed: " #condition); \
        } \
    } while(0)

#define ASSERT_FALSE(condition) \
    do { \
        if (condition) { \
            throw std::runtime_error("Assertion failed: !(" #condition ")"); \
        } \
    } while(0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if ((expected) != (actual)) { \
            throw std::runtime_error("Assertion failed: " #expected " == " #actual \
                " (expected: " + std::to_string(expected) + ", actual: " + std::to_string(actual) + ")"); \
        } \
    } while(0)

#define ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        if (std::abs((expected) - (actual)) > (tolerance)) { \
            throw std::runtime_error("Assertion failed: " #expected " ≈ " #actual " ± " #tolerance \
                " (expected: " + std::to_string(expected) + ", actual: " + std::to_string(actual) + \
                ", diff: " + std::to_string(std::abs((expected) - (actual))) + ")"); \
        } \
    } while(0)

#define ASSERT_GT(value, threshold) \
    do { \
        if ((value) <= (threshold)) { \
            throw std::runtime_error("Assertion failed: " #value " > " #threshold \
                " (value: " + std::to_string(value) + ", threshold: " + std::to_string(threshold) + ")"); \
        } \
    } while(0)

#define ASSERT_LT(value, threshold) \
    do { \
        if ((value) >= (threshold)) { \
            throw std::runtime_error("Assertion failed: " #value " < " #threshold \
                " (value: " + std::to_string(value) + ", threshold: " + std::to_string(threshold) + ")"); \
        } \
    } while(0)

#define ASSERT_GE(value, threshold) \
    do { \
        if ((value) < (threshold)) { \
            throw std::runtime_error("Assertion failed: " #value " >= " #threshold \
                " (value: " + std::to_string(value) + ", threshold: " + std::to_string(threshold) + ")"); \
        } \
    } while(0)

#define ASSERT_LE(value, threshold) \
    do { \
        if ((value) > (threshold)) { \
            throw std::runtime_error("Assertion failed: " #value " <= " #threshold \
                " (value: " + std::to_string(value) + ", threshold: " + std::to_string(threshold) + ")"); \
        } \
    } while(0)

// Test registration macro
#define TEST(category, name) \
    void test_##category##_##name(); \
    static bool registered_##category##_##name = []() { \
        TestFramework::addTest(#name, #category, test_##category##_##name); \
        return true; \
    }(); \
    void test_##category##_##name()

// Static member definitions
std::vector<TestFramework::Test> TestFramework::tests_;
int TestFramework::passed_ = 0;
int TestFramework::failed_ = 0;
std::string TestFramework::current_test_;

#endif // TEST_FRAMEWORK_H 