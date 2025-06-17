#include "rebound_streaming.h"
#include "rebound_utils.h"
#include <iostream>
#include <cstring>
#include <algorithm>

// =============================================================================
// GPUDataBuffer Implementation
// =============================================================================

GPUDataBuffer::GPUDataBuffer(int max_frames, int max_logs) 
    : d_frames_(nullptr), d_logs_(nullptr), d_frame_write_idx_(nullptr), 
      d_log_write_idx_(nullptr), max_frames_(max_frames), max_logs_(max_logs), 
      allocated_(false) {
}

GPUDataBuffer::~GPUDataBuffer() {
    deallocate();
}

bool GPUDataBuffer::allocate() {
    if (allocated_) return true;
    
    cudaError_t err;
    
    // Allocate frame buffer
    err = cudaMalloc(&d_frames_, max_frames_ * sizeof(SimulationFrame));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU frame buffer: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    
    // Allocate log buffer
    err = cudaMalloc(&d_logs_, max_logs_ * sizeof(GPULogEntry));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate GPU log buffer: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_frames_);
        return false;
    }
    
    // Allocate write indices
    err = cudaMalloc(&d_frame_write_idx_, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate frame write index: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_frames_);
        cudaFree(d_logs_);
        return false;
    }
    
    err = cudaMalloc(&d_log_write_idx_, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate log write index: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_frames_);
        cudaFree(d_logs_);
        cudaFree(d_frame_write_idx_);
        return false;
    }
    
    // Initialize indices to 0
    int zero = 0;
    cudaMemcpy(d_frame_write_idx_, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_write_idx_, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    allocated_ = true;
    return true;
}

void GPUDataBuffer::deallocate() {
    if (!allocated_) return;
    
    if (d_frames_) cudaFree(d_frames_);
    if (d_logs_) cudaFree(d_logs_);
    if (d_frame_write_idx_) cudaFree(d_frame_write_idx_);
    if (d_log_write_idx_) cudaFree(d_log_write_idx_);
    
    d_frames_ = nullptr;
    d_logs_ = nullptr;
    d_frame_write_idx_ = nullptr;
    d_log_write_idx_ = nullptr;
    allocated_ = false;
}

int GPUDataBuffer::getCurrentFrameCount() const {
    if (!allocated_) return 0;
    
    int count;
    cudaMemcpy(&count, d_frame_write_idx_, sizeof(int), cudaMemcpyDeviceToHost);
    return std::min(count, max_frames_);
}

int GPUDataBuffer::getCurrentLogCount() const {
    if (!allocated_) return 0;
    
    int count;
    cudaMemcpy(&count, d_log_write_idx_, sizeof(int), cudaMemcpyDeviceToHost);
    return std::min(count, max_logs_);
}

void GPUDataBuffer::resetBuffers() {
    if (!allocated_) return;
    
    int zero = 0;
    cudaMemcpy(d_frame_write_idx_, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_log_write_idx_, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

void GPUDataBuffer::copyFramesToHost(SimulationFrame* host_frames, int count) {
    if (!allocated_ || !host_frames || count <= 0) return;
    
    int available_frames = std::min(count, getCurrentFrameCount());
    if (available_frames > 0) {
        cudaMemcpy(host_frames, d_frames_, available_frames * sizeof(SimulationFrame), 
                  cudaMemcpyDeviceToHost);
    }
}

void GPUDataBuffer::copyLogsToHost(GPULogEntry* host_logs, int count) {
    if (!allocated_ || !host_logs || count <= 0) return;
    
    int available_logs = std::min(count, getCurrentLogCount());
    if (available_logs > 0) {
        cudaMemcpy(host_logs, d_logs_, available_logs * sizeof(GPULogEntry), 
                  cudaMemcpyDeviceToHost);
    }
}

// Device-side functions
__device__ void GPUDataBuffer::logFrame(double time, int step, Particle* particles, 
                                       int n_particles, double energy, int n_collisions) {
    int idx = atomicAdd(d_frame_write_idx_, 1);
    if (idx >= max_frames_) return; // Buffer full
    
    SimulationFrame& frame = d_frames_[idx % max_frames_];
    frame.time = time;
    frame.step = step;
    frame.n_particles = n_particles;
    frame.particles = particles; // Store device pointer
    frame.total_energy = energy;
    frame.n_collisions = n_collisions;
}

__device__ void GPUDataBuffer::logMessage(double time, int step, LogLevel level, 
                                         const char* format, ...) {
    int idx = atomicAdd(d_log_write_idx_, 1);
    if (idx >= max_logs_) return; // Buffer full
    
    GPULogEntry& entry = d_logs_[idx % max_logs_];
    entry.time = time;
    entry.step = step;
    entry.level = level;
    entry.thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Simple string copy (no printf formatting on device for now)
    int i = 0;
    while (format[i] != '\0' && i < 255) {
        entry.message[i] = format[i];
        i++;
    }
    entry.message[i] = '\0';
}

// =============================================================================
// AsyncProcessor Implementation
// =============================================================================

AsyncProcessor::AsyncProcessor(int num_threads) 
    : running_(false), frame_callback_(nullptr), log_callback_(nullptr),
      frame_user_data_(nullptr), log_user_data_(nullptr) {
    worker_threads_.reserve(num_threads);
}

AsyncProcessor::~AsyncProcessor() {
    stop();
}

void AsyncProcessor::start() {
    if (running_) return;
    
    running_ = true;
    
    // Start worker threads
    for (size_t i = 0; i < worker_threads_.capacity(); ++i) {
        worker_threads_.emplace_back(&AsyncProcessor::workerLoop, this);
    }
}

void AsyncProcessor::stop() {
    if (!running_) return;
    
    running_ = false;
    
    // Wake up all threads
    frame_cv_.notify_all();
    log_cv_.notify_all();
    
    // Join all threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    worker_threads_.clear();
}

void AsyncProcessor::setFrameCallback(FrameCallback callback, void* user_data) {
    frame_callback_ = callback;
    frame_user_data_ = user_data;
}

void AsyncProcessor::setLogCallback(LogCallback callback, void* user_data) {
    log_callback_ = callback;
    log_user_data_ = user_data;
}

void AsyncProcessor::enqueueFrame(const SimulationFrame& frame) {
    auto frame_copy = std::make_unique<SimulationFrame>(frame);
    
    std::lock_guard<std::mutex> lock(frame_mutex_);
    frame_queue_.push(std::move(frame_copy));
    frame_cv_.notify_one();
}

void AsyncProcessor::enqueueLog(const GPULogEntry& log) {
    auto log_copy = std::make_unique<GPULogEntry>(log);
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    log_queue_.push(std::move(log_copy));
    log_cv_.notify_one();
}

void AsyncProcessor::workerLoop() {
    while (running_) {
        // Process frames
        {
            std::unique_lock<std::mutex> lock(frame_mutex_);
            frame_cv_.wait(lock, [this] { return !frame_queue_.empty() || !running_; });
            
            while (!frame_queue_.empty()) {
                auto frame = std::move(frame_queue_.front());
                frame_queue_.pop();
                lock.unlock();
                
                if (frame_callback_) {
                    frame_callback_(frame.get(), frame_user_data_);
                }
                
                lock.lock();
            }
        }
        
        // Process logs
        {
            std::unique_lock<std::mutex> lock(log_mutex_);
            log_cv_.wait(lock, [this] { return !log_queue_.empty() || !running_; });
            
            while (!log_queue_.empty()) {
                auto log = std::move(log_queue_.front());
                log_queue_.pop();
                lock.unlock();
                
                if (log_callback_) {
                    log_callback_(log.get(), log_user_data_);
                }
                
                lock.lock();
            }
        }
    }
}

size_t AsyncProcessor::getFrameQueueSize() const {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    return frame_queue_.size();
}

size_t AsyncProcessor::getLogQueueSize() const {
    std::lock_guard<std::mutex> lock(log_mutex_);
    return log_queue_.size();
}

// =============================================================================
// DataStreamingManager Implementation
// =============================================================================

DataStreamingManager::DataStreamingManager() 
    : mode_(STREAM_NONE), current_step_(0), last_streamed_step_(-1), 
      initialized_(false), data_stream_(0), log_stream_(0), cached_d_particles_(nullptr) {
}

DataStreamingManager::~DataStreamingManager() {
    shutdown();
}

bool DataStreamingManager::initialize(const BufferConfig& config) {
    if (initialized_) return true;
    
    config_ = config;
    
    // Allocate GPU buffers
    if (!gpu_buffer_.allocate()) {
        return false;
    }
    
    // Initialize async processor if needed
    if (config_.async_streaming) {
        async_processor_ = std::make_unique<AsyncProcessor>(config_.worker_threads);
        async_processor_->start();
    }
    
    // Create CUDA streams for async operations
    cudaStreamCreate(&data_stream_);
    cudaStreamCreate(&log_stream_);
    
    // Pre-allocate host buffers
    h_frame_buffer_.reserve(config_.max_frames);
    h_log_buffer_.reserve(10000);
    
    initialized_ = true;
    return true;
}

void DataStreamingManager::shutdown() {
    if (!initialized_) return;
    
    if (async_processor_) {
        async_processor_->stop();
        async_processor_.reset();
    }
    
    gpu_buffer_.deallocate();
    
    if (data_stream_) cudaStreamDestroy(data_stream_);
    if (log_stream_) cudaStreamDestroy(log_stream_);
    
    initialized_ = false;
}

void DataStreamingManager::setStreamingMode(StreamingMode mode) {
    mode_ = mode;
}

void DataStreamingManager::setFrameCallback(FrameCallback callback, void* user_data) {
    if (async_processor_) {
        async_processor_->setFrameCallback(callback, user_data);
    }
}

void DataStreamingManager::setLogCallback(LogCallback callback, void* user_data) {
    if (async_processor_) {
        async_processor_->setLogCallback(callback, user_data);
    }
}

// =============================================================================
// SimulationObserver Interface Implementation
// =============================================================================

void DataStreamingManager::onSimulationStart(int n_particles) {
    std::cout << "StreamingManager: Simulation started with " << n_particles << " particles" << std::endl;
    current_step_ = 0;
    last_streamed_step_ = -1;
    
    if (initialized_) {
        gpu_buffer_.resetBuffers();
    }
}

void DataStreamingManager::onSimulationStep(double time, int step, int n_particles, double energy, int n_collisions) {
    if (!initialized_) return;
    
    current_step_ = step;
    
    // Determine if we should stream this step
    bool should_stream = false;
    
    switch (mode_) {
        case STREAM_NONE:
            should_stream = false;
            break;
        case STREAM_CONTINUOUS:
            should_stream = true;
            break;
        case STREAM_PERIODIC:
            should_stream = (step % config_.stream_interval == 0);
            break;
        case STREAM_ON_DEMAND:
            should_stream = false; // Only when explicitly requested
            break;
    }
    
    if (should_stream) {
        streamCurrentData();
        std::cout << "Streamed data at step " << step << ", time " << time << std::endl;
    }
}

void DataStreamingManager::onSimulationEnd(double final_time, int total_steps) {
    std::cout << "StreamingManager: Simulation ended at time " << final_time 
              << " after " << total_steps << " steps" << std::endl;
    
    if (initialized_) {
        std::cout << "Final buffered frames: " << getBufferedFrameCount() << std::endl;
        std::cout << "Final buffered logs: " << getBufferedLogCount() << std::endl;
    }
}

void DataStreamingManager::streamCurrentData() {
    if (!initialized_) return;
    
    // This would trigger GPU kernels to log current frame
    // Implementation depends on integration with simulation loop
    last_streamed_step_ = current_step_;
}

std::vector<SimulationFrame> DataStreamingManager::getAllFrames() {
    if (!initialized_) return {};
    
    int frame_count = gpu_buffer_.getCurrentFrameCount();
    std::vector<SimulationFrame> frames(frame_count);
    
    if (frame_count > 0) {
        gpu_buffer_.copyFramesToHost(frames.data(), frame_count);
    }
    
    return frames;
}

std::vector<GPULogEntry> DataStreamingManager::getAllLogs() {
    if (!initialized_) return {};
    
    int log_count = gpu_buffer_.getCurrentLogCount();
    std::vector<GPULogEntry> logs(log_count);
    
    if (log_count > 0) {
        gpu_buffer_.copyLogsToHost(logs.data(), log_count);
    }
    
    return logs;
}

int DataStreamingManager::getBufferedFrameCount() const {
    return initialized_ ? gpu_buffer_.getCurrentFrameCount() : 0;
}

int DataStreamingManager::getBufferedLogCount() const {
    return initialized_ ? gpu_buffer_.getCurrentLogCount() : 0;
}

size_t DataStreamingManager::getAsyncQueueSize() const {
    return async_processor_ ? async_processor_->getFrameQueueSize() : 0;
}

// =============================================================================
// GPU Logging Functions
// =============================================================================

__device__ void gpu_log_error(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...) {
    if (gpu_buffer) {
        gpu_buffer->logMessage(time, step, LOG_ERROR, format);
    }
}

__device__ void gpu_log_warning(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...) {
    if (gpu_buffer) {
        gpu_buffer->logMessage(time, step, LOG_WARNING, format);
    }
}

__device__ void gpu_log_info(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...) {
    if (gpu_buffer) {
        gpu_buffer->logMessage(time, step, LOG_INFO, format);
    }
}

__device__ void gpu_log_debug(GPUDataBuffer* gpu_buffer, double time, int step, const char* format, ...) {
    if (gpu_buffer) {
        gpu_buffer->logMessage(time, step, LOG_DEBUG, format);
    }
} 