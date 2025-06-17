#ifndef REBOUND_STREAMING_H
#define REBOUND_STREAMING_H

#include "rebound_types.h"
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <memory>

// Forward declarations
class AsyncProcessor;

// GPU-side data buffer manager (ring buffer on GPU)
class GPUDataBuffer {
private:
    SimulationFrame* d_frames_;     // Device memory ring buffer
    GPULogEntry* d_logs_;          // Device memory log buffer
    int* d_frame_write_idx_;       // Current write index for frames
    int* d_log_write_idx_;         // Current write index for logs
    int max_frames_;
    int max_logs_;
    bool allocated_;

public:
    GPUDataBuffer(int max_frames = 1000, int max_logs = 10000);
    ~GPUDataBuffer();
    
    bool allocate();
    void deallocate();
    
    // Device-side functions (called from kernels)
    __device__ void logFrame(double time, int step, Particle* particles, int n_particles, 
                            double energy, int n_collisions);
    __device__ void logMessage(double time, int step, LogLevel level, const char* format, ...);
    
    // Host-side access
    int getCurrentFrameCount() const;
    int getCurrentLogCount() const;
    void copyFramesToHost(SimulationFrame* h_frames, int count);
    void copyLogsToHost(GPULogEntry* h_logs, int count);
    void resetBuffers();
};

// Host-side async data processor
class AsyncProcessor {
private:
    std::queue<std::unique_ptr<SimulationFrame>> frame_queue_;
    std::queue<std::unique_ptr<GPULogEntry>> log_queue_;
    mutable std::mutex frame_mutex_;
    mutable std::mutex log_mutex_;
    std::condition_variable frame_cv_;
    std::condition_variable log_cv_;
    std::atomic<bool> running_;
    std::vector<std::thread> worker_threads_;
    
    FrameCallback frame_callback_;
    LogCallback log_callback_;
    void* frame_user_data_;
    void* log_user_data_;
    
    void workerLoop();
    
public:
    AsyncProcessor(int num_threads = 2);
    ~AsyncProcessor();
    
    void start();
    void stop();
    
    void setFrameCallback(FrameCallback callback, void* user_data);
    void setLogCallback(LogCallback callback, void* user_data);
    
    void enqueueFrame(const SimulationFrame& frame);
    void enqueueLog(const GPULogEntry& log);
    
    size_t getFrameQueueSize() const;
    size_t getLogQueueSize() const;
};

// Main data streaming manager - now an optional observer
class DataStreamingManager : public SimulationObserver {
private:
    StreamingMode mode_;
    BufferConfig config_;
    GPUDataBuffer gpu_buffer_;
    std::unique_ptr<AsyncProcessor> async_processor_;
    
    // Host-side buffers for batch operations
    std::vector<SimulationFrame> h_frame_buffer_;
    std::vector<GPULogEntry> h_log_buffer_;
    
    // State tracking
    int current_step_;
    int last_streamed_step_;
    bool initialized_;
    
    // CUDA streams for async operations
    cudaStream_t data_stream_;
    cudaStream_t log_stream_;
    
    // Device pointer cache (received from simulation)
    Particle* cached_d_particles_;
    
public:
    DataStreamingManager();
    ~DataStreamingManager();
    
    bool initialize(const BufferConfig& config);
    void shutdown();
    
    // Configuration
    void setStreamingMode(StreamingMode mode);
    void setFrameCallback(FrameCallback callback, void* user_data);
    void setLogCallback(LogCallback callback, void* user_data);
    
    // SimulationObserver interface implementation
    void onSimulationStep(double time, int step, int n_particles, double energy = 0.0, int n_collisions = 0) override;
    void onSimulationStart(int n_particles) override;
    void onSimulationEnd(double final_time, int total_steps) override;
    
    // Manual data access (for STREAM_ON_DEMAND and end-of-simulation)
    void streamCurrentData();
    std::vector<SimulationFrame> getAllFrames();
    std::vector<GPULogEntry> getAllLogs();
    
    // Statistics
    int getBufferedFrameCount() const;
    int getBufferedLogCount() const;
    size_t getAsyncQueueSize() const;
    
    // GPU logger interface for kernels
    GPUDataBuffer* getGPUBuffer() { return &gpu_buffer_; }
    
    // Internal method to cache device particle pointer from simulation
    void setDeviceParticlePointer(Particle* d_particles) { cached_d_particles_ = d_particles; }
};

// CUDA device functions for logging (to be used in kernels)
__device__ void gpu_log_error(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...);
__device__ void gpu_log_warning(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...);
__device__ void gpu_log_info(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...);
__device__ void gpu_log_debug(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...);

// Convenience macros for GPU logging (need to be passed gpu_buffer pointer)
#define GPU_LOG_ERROR(gpu_buffer, time, step, ...) gpu_log_error(gpu_buffer, time, step, __VA_ARGS__)
#define GPU_LOG_WARNING(gpu_buffer, time, step, ...) gpu_log_warning(gpu_buffer, time, step, __VA_ARGS__)
#define GPU_LOG_INFO(gpu_buffer, time, step, ...) gpu_log_info(gpu_buffer, time, step, __VA_ARGS__)
#define GPU_LOG_DEBUG(gpu_buffer, time, step, ...) gpu_log_debug(gpu_buffer, time, step, __VA_ARGS__)

#endif // REBOUND_STREAMING_H 