#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {
void copy_message(char *buffer, size_t capacity, NSString *message) {
    if (buffer == nullptr || capacity == 0) return;
    const char *source = message.UTF8String ?: "unknown Metal error";
    std::strncpy(buffer, source, capacity - 1);
    buffer[capacity - 1] = '\0';
}

struct CachedGraph {
    MPSGraph *graph;
    MPSGraphTensor *input;
    MPSGraphTensor *output;
    MPSShape *shape;
};

struct CachedLocalGraph {
    MPSGraph *graph;
    MPSGraphTensor *input;
    MPSGraphTensor *potential;
    MPSGraphTensor *kinetic;
    MPSGraphTensor *output;
    MPSShape *input_shape;
    MPSShape *potential_shape;
};

struct CachedMatrixGraph {
    MPSGraph *graph;
    MPSGraphTensor *input;
    MPSGraphTensor *matrix;
    MPSGraphTensor *output;
    MPSShape *input_shape;
    MPSShape *matrix_shape;
};

std::mutex cache_mutex;
std::unordered_map<std::string, CachedGraph> graph_cache;
std::unordered_map<std::string, CachedLocalGraph> local_graph_cache;
std::unordered_map<std::string, CachedMatrixGraph> matrix_graph_cache;

id<MTLDevice> shared_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> shared_queue() {
    static id<MTLCommandQueue> queue = [shared_device() newCommandQueue];
    return queue;
}

CachedGraph graph_for(size_t nx, size_t ny, size_t nz, size_t batch) {
    std::string key = std::to_string(batch) + ":" + std::to_string(nx) + ":" +
                      std::to_string(ny) + ":" + std::to_string(nz);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = graph_cache.find(key);
    if (found != graph_cache.end()) return found->second;

    MPSShape *shape = @[@(batch), @(nx), @(ny), @(nz)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *input = [graph placeholderWithShape:shape
                                               dataType:MPSDataTypeComplexFloat32
                                                   name:@"wavefunctions"];
    MPSGraphFFTDescriptor *forward_desc = [MPSGraphFFTDescriptor descriptor];
    forward_desc.inverse = NO;
    forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor *frequency = [graph fastFourierTransformWithTensor:input
                                                                 axes:@[@1, @2, @3]
                                                           descriptor:forward_desc
                                                                 name:@"forward"];
    MPSGraphFFTDescriptor *inverse_desc = [MPSGraphFFTDescriptor descriptor];
    inverse_desc.inverse = YES;
    inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
    MPSGraphTensor *output = [graph fastFourierTransformWithTensor:frequency
                                                              axes:@[@1, @2, @3]
                                                        descriptor:inverse_desc
                                                              name:@"inverse"];
    CachedGraph cached{graph, input, output, shape};
    graph_cache.emplace(key, cached);
    return cached;
}

CachedLocalGraph local_graph_for(size_t nx, size_t ny, size_t nz, size_t batch) {
    std::string key = std::to_string(batch) + ":" + std::to_string(nx) + ":" +
                      std::to_string(ny) + ":" + std::to_string(nz);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = local_graph_cache.find(key);
    if (found != local_graph_cache.end()) return found->second;

    MPSShape *input_shape = @[@(batch), @(nx), @(ny), @(nz)];
    MPSShape *potential_shape = @[@1, @(nx), @(ny), @(nz)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *input = [graph placeholderWithShape:input_shape
                                               dataType:MPSDataTypeComplexFloat32
                                                   name:@"reciprocal_wavefunctions"];
    MPSGraphTensor *potential = [graph placeholderWithShape:potential_shape
                                                   dataType:MPSDataTypeComplexFloat32
                                                       name:@"local_potential"];
    MPSGraphTensor *kinetic = [graph placeholderWithShape:potential_shape
                                                 dataType:MPSDataTypeComplexFloat32
                                                     name:@"kinetic_energy"];
    MPSGraphFFTDescriptor *inverse_desc = [MPSGraphFFTDescriptor descriptor];
    inverse_desc.inverse = YES;
    inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
    MPSGraphTensor *real_space = [graph fastFourierTransformWithTensor:input
                                                                        axes:@[@1, @2, @3]
                                                                  descriptor:inverse_desc
                                                                        name:@"inverse"];
    MPSGraphTensor *multiplied = [graph multiplicationWithPrimaryTensor:real_space
                                                              secondaryTensor:potential
                                                                         name:@"local_multiply"];
    MPSGraphFFTDescriptor *forward_desc = [MPSGraphFFTDescriptor descriptor];
    forward_desc.inverse = NO;
    forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor *local_output = [graph fastFourierTransformWithTensor:multiplied
                                                                          axes:@[@1, @2, @3]
                                                                    descriptor:forward_desc
                                                                          name:@"forward"];
    MPSGraphTensor *kinetic_output = [graph multiplicationWithPrimaryTensor:input
                                                                  secondaryTensor:kinetic
                                                                             name:@"kinetic_multiply"];
    MPSGraphTensor *output = [graph additionWithPrimaryTensor:local_output
                                                secondaryTensor:kinetic_output
                                                           name:@"hamiltonian_sum"];
    CachedLocalGraph cached{graph, input, potential, kinetic, output, input_shape, potential_shape};
    local_graph_cache.emplace(key, cached);
    return cached;
}

CachedMatrixGraph matrix_graph_for(size_t dimension, size_t batch) {
    std::string key = std::to_string(batch) + ":" + std::to_string(dimension);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = matrix_graph_cache.find(key);
    if (found != matrix_graph_cache.end()) return found->second;

    MPSShape *input_shape = @[@(batch), @(dimension)];
    MPSShape *matrix_shape = @[@(dimension), @(dimension)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *input = [graph placeholderWithShape:input_shape
                                               dataType:MPSDataTypeComplexFloat32
                                                   name:@"vectors"];
    MPSGraphTensor *matrix = [graph placeholderWithShape:matrix_shape
                                                dataType:MPSDataTypeComplexFloat32
                                                    name:@"operator_transpose"];
    MPSGraphTensor *output = [graph matrixMultiplicationWithPrimaryTensor:input
                                                             secondaryTensor:matrix
                                                                        name:@"dense_operator"];
    CachedMatrixGraph cached{graph, input, matrix, output, input_shape, matrix_shape};
    matrix_graph_cache.emplace(key, cached);
    return cached;
}
}  // namespace

extern "C" int rpd_metal_device_name(char *buffer, size_t capacity) {
    @autoreleasepool {
        id<MTLDevice> device = shared_device();
        if (device == nil) {
            copy_message(buffer, capacity, @"no Metal device");
            return 1;
        }
        copy_message(buffer, capacity, device.name);
        return 0;
    }
}

extern "C" int rpd_metal_fft_roundtrip(const float *input_interleaved,
                                         float *output_interleaved,
                                         size_t nx, size_t ny, size_t nz,
                                         size_t batch,
                                         char *error_buffer,
                                         size_t error_capacity) {
    @autoreleasepool {
        if (!input_interleaved || !output_interleaved || nx == 0 || ny == 0 || nz == 0 || batch == 0) {
            copy_message(error_buffer, error_capacity, @"invalid FFT arguments");
            return 2;
        }
        id<MTLDevice> device = shared_device();
        id<MTLCommandQueue> queue = shared_queue();
        if (device == nil || queue == nil) {
            copy_message(error_buffer, error_capacity, @"failed to create Metal device or queue");
            return 3;
        }
        if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / batch) {
            copy_message(error_buffer, error_capacity, @"FFT allocation overflow");
            return 4;
        }
        const size_t complex_count = nx * ny * nz * batch;
        const size_t bytes = complex_count * 2 * sizeof(float);
        if (bytes > std::min((uint64_t)(12ULL << 30), device.recommendedMaxWorkingSetSize)) {
            copy_message(error_buffer, error_capacity, @"FFT request exceeds the 12 GiB project limit");
            return 5;
        }
        id<MTLBuffer> buffer = [device newBufferWithBytes:input_interleaved
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            copy_message(error_buffer, error_capacity, @"failed to allocate shared MTLBuffer");
            return 6;
        }
        CachedGraph cached = graph_for(nx, ny, nz, batch);
        MPSGraphTensorData *data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                         shape:cached.shape
                                                                      dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:queue
                                                                  feeds:@{cached.input: data}
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:output_interleaved strideBytes:nil];
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 7;
        }
        return 0;
    }
}

extern "C" int rpd_metal_apply_local(const float *input_interleaved,
                                      const float *potential_interleaved,
                                      const float *kinetic_interleaved,
                                      float *output_interleaved,
                                      size_t nx, size_t ny, size_t nz,
                                      size_t batch,
                                      char *error_buffer,
                                      size_t error_capacity) {
    @autoreleasepool {
        if (!input_interleaved || !potential_interleaved || !kinetic_interleaved ||
            !output_interleaved ||
            nx == 0 || ny == 0 || nz == 0 || batch == 0) {
            copy_message(error_buffer, error_capacity, @"invalid local-operator arguments");
            return 2;
        }
        id<MTLDevice> device = shared_device();
        id<MTLCommandQueue> queue = shared_queue();
        if (device == nil || queue == nil) {
            copy_message(error_buffer, error_capacity, @"failed to create Metal device or queue");
            return 3;
        }
        if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz || nx * ny * nz > SIZE_MAX / batch) {
            copy_message(error_buffer, error_capacity, @"local-operator allocation overflow");
            return 4;
        }
        const size_t grid_count = nx * ny * nz;
        const size_t complex_count = grid_count * batch;
        const size_t input_bytes = complex_count * 2 * sizeof(float);
        const size_t potential_bytes = grid_count * 2 * sizeof(float);
        const uint64_t project_limit = std::min((uint64_t)(12ULL << 30),
                                                device.recommendedMaxWorkingSetSize);
        if (input_bytes > project_limit || potential_bytes > project_limit - input_bytes ||
            potential_bytes > project_limit - input_bytes - potential_bytes) {
            copy_message(error_buffer, error_capacity,
                         @"local-operator request exceeds the 12 GiB project limit");
            return 5;
        }
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_interleaved
                                                         length:input_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> potential_buffer = [device newBufferWithBytes:potential_interleaved
                                                             length:potential_bytes
                                                            options:MTLResourceStorageModeShared];
        id<MTLBuffer> kinetic_buffer = [device newBufferWithBytes:kinetic_interleaved
                                                           length:potential_bytes
                                                          options:MTLResourceStorageModeShared];
        if (input_buffer == nil || potential_buffer == nil || kinetic_buffer == nil) {
            copy_message(error_buffer, error_capacity, @"failed to allocate shared MTLBuffers");
            return 6;
        }
        CachedLocalGraph cached = local_graph_for(nx, ny, nz, batch);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input_buffer
                        shape:cached.input_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *potential_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:potential_buffer
                        shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *kinetic_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:kinetic_buffer
                        shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:queue
                                                                  feeds:@{
                                                                      cached.input: input_data,
                                                                      cached.potential: potential_data,
                                                                      cached.kinetic: kinetic_data
                                                                  }
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:output_interleaved strideBytes:nil];
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 7;
        }
        return 0;
    }
}

extern "C" int rpd_metal_apply_dense(const float *input_interleaved,
                                      const float *matrix_interleaved,
                                      float *output_interleaved,
                                      size_t dimension,
                                      size_t batch,
                                      char *error_buffer,
                                      size_t error_capacity) {
    @autoreleasepool {
        if (!input_interleaved || !matrix_interleaved || !output_interleaved ||
            dimension == 0 || batch == 0) {
            copy_message(error_buffer, error_capacity, @"invalid dense-operator arguments");
            return 2;
        }
        id<MTLDevice> device = shared_device();
        id<MTLCommandQueue> queue = shared_queue();
        if (device == nil || queue == nil) {
            copy_message(error_buffer, error_capacity, @"failed to create Metal device or queue");
            return 3;
        }
        if (dimension > SIZE_MAX / dimension || dimension > SIZE_MAX / batch) {
            copy_message(error_buffer, error_capacity, @"dense-operator allocation overflow");
            return 4;
        }
        const size_t input_count = dimension * batch;
        const size_t matrix_count = dimension * dimension;
        const size_t input_bytes = input_count * 2 * sizeof(float);
        const size_t matrix_bytes = matrix_count * 2 * sizeof(float);
        const uint64_t project_limit = std::min((uint64_t)(12ULL << 30),
                                                device.recommendedMaxWorkingSetSize);
        if (input_bytes > project_limit || matrix_bytes > project_limit - input_bytes) {
            copy_message(error_buffer, error_capacity,
                         @"dense-operator request exceeds the 12 GiB project limit");
            return 5;
        }
        id<MTLBuffer> input_buffer = [device newBufferWithBytes:input_interleaved
                                                         length:input_bytes
                                                        options:MTLResourceStorageModeShared];
        id<MTLBuffer> matrix_buffer = [device newBufferWithBytes:matrix_interleaved
                                                          length:matrix_bytes
                                                         options:MTLResourceStorageModeShared];
        if (input_buffer == nil || matrix_buffer == nil) {
            copy_message(error_buffer, error_capacity, @"failed to allocate shared MTLBuffers");
            return 6;
        }
        CachedMatrixGraph cached = matrix_graph_for(dimension, batch);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:input_buffer
                        shape:cached.input_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *matrix_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:matrix_buffer
                        shape:cached.matrix_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:queue
                                                                  feeds:@{
                                                                      cached.input: input_data,
                                                                      cached.matrix: matrix_data
                                                                  }
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:output_interleaved strideBytes:nil];
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 7;
        }
        return 0;
    }
}
