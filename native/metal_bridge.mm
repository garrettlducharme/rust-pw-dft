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

struct CachedResidentGraph {
    MPSGraph *graph;
    MPSGraphTensor *grid_input;
    MPSGraphTensor *compact_input;
    MPSGraphTensor *potential;
    MPSGraphTensor *kinetic;
    MPSGraphTensor *projector_conjugate;
    MPSGraphTensor *d_transpose;
    MPSGraphTensor *projector_transpose;
    MPSGraphTensor *local_output;
    MPSGraphTensor *nonlocal_output;
    MPSShape *grid_shape;
    MPSShape *compact_shape;
    MPSShape *potential_shape;
    MPSShape *projector_conjugate_shape;
    MPSShape *d_shape;
    MPSShape *projector_transpose_shape;
};

struct ResidentHamiltonian {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    size_t nx;
    size_t ny;
    size_t nz;
    size_t grid_count;
    size_t dimension;
    size_t projector_count;
    size_t max_batch;
    uint64_t peak_device_allocated_bytes;
    id<MTLBuffer> grid_input;
    id<MTLBuffer> compact_input;
    id<MTLBuffer> potential;
    id<MTLBuffer> kinetic;
    id<MTLBuffer> projector_conjugate;
    id<MTLBuffer> d_transpose;
    id<MTLBuffer> projector_transpose;
};

std::mutex cache_mutex;
std::unordered_map<std::string, CachedGraph> graph_cache;
std::unordered_map<std::string, CachedLocalGraph> local_graph_cache;
std::unordered_map<std::string, CachedMatrixGraph> matrix_graph_cache;
std::unordered_map<std::string, CachedResidentGraph> resident_graph_cache;

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

CachedResidentGraph resident_graph_for(size_t nx, size_t ny, size_t nz,
                                       size_t dimension, size_t projector_count,
                                       size_t batch) {
    std::string key = std::to_string(batch) + ":" + std::to_string(nx) + ":" +
                      std::to_string(ny) + ":" + std::to_string(nz) + ":" +
                      std::to_string(dimension) + ":" + std::to_string(projector_count);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = resident_graph_cache.find(key);
    if (found != resident_graph_cache.end()) return found->second;

    MPSShape *grid_shape = @[@(batch), @(nx), @(ny), @(nz)];
    MPSShape *compact_shape = @[@(batch), @(dimension)];
    MPSShape *potential_shape = @[@1, @(nx), @(ny), @(nz)];
    MPSShape *projector_conjugate_shape = @[@(dimension), @(projector_count)];
    MPSShape *d_shape = @[@(projector_count), @(projector_count)];
    MPSShape *projector_transpose_shape = @[@(projector_count), @(dimension)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *grid_input = [graph placeholderWithShape:grid_shape
                                                      dataType:MPSDataTypeComplexFloat32
                                                          name:@"grid_wavefunctions"];
    MPSGraphTensor *compact_input = [graph placeholderWithShape:compact_shape
                                                         dataType:MPSDataTypeComplexFloat32
                                                             name:@"compact_wavefunctions"];
    MPSGraphTensor *potential = [graph placeholderWithShape:potential_shape
                                                   dataType:MPSDataTypeComplexFloat32
                                                       name:@"local_potential"];
    MPSGraphTensor *kinetic = [graph placeholderWithShape:potential_shape
                                                 dataType:MPSDataTypeComplexFloat32
                                                     name:@"kinetic_energy"];
    MPSGraphTensor *projector_conjugate =
        [graph placeholderWithShape:projector_conjugate_shape
                           dataType:MPSDataTypeComplexFloat32
                               name:@"projector_conjugate"];
    MPSGraphTensor *d_transpose = [graph placeholderWithShape:d_shape
                                                     dataType:MPSDataTypeComplexFloat32
                                                         name:@"d_transpose"];
    MPSGraphTensor *projector_transpose =
        [graph placeholderWithShape:projector_transpose_shape
                           dataType:MPSDataTypeComplexFloat32
                               name:@"projector_transpose"];

    MPSGraphFFTDescriptor *inverse_desc = [MPSGraphFFTDescriptor descriptor];
    inverse_desc.inverse = YES;
    inverse_desc.scalingMode = MPSGraphFFTScalingModeSize;
    MPSGraphTensor *real_space = [graph fastFourierTransformWithTensor:grid_input
                                                                        axes:@[@1, @2, @3]
                                                                  descriptor:inverse_desc
                                                                        name:@"inverse"];
    MPSGraphTensor *multiplied = [graph multiplicationWithPrimaryTensor:real_space
                                                              secondaryTensor:potential
                                                                         name:@"local_multiply"];
    MPSGraphFFTDescriptor *forward_desc = [MPSGraphFFTDescriptor descriptor];
    forward_desc.inverse = NO;
    forward_desc.scalingMode = MPSGraphFFTScalingModeNone;
    MPSGraphTensor *local_fft = [graph fastFourierTransformWithTensor:multiplied
                                                                        axes:@[@1, @2, @3]
                                                                  descriptor:forward_desc
                                                                        name:@"forward"];
    MPSGraphTensor *kinetic_output = [graph multiplicationWithPrimaryTensor:grid_input
                                                                  secondaryTensor:kinetic
                                                                             name:@"kinetic_multiply"];
    MPSGraphTensor *local_output = [graph additionWithPrimaryTensor:local_fft
                                                    secondaryTensor:kinetic_output
                                                               name:@"local_kinetic_sum"];
    MPSGraphTensor *overlaps = [graph matrixMultiplicationWithPrimaryTensor:compact_input
                                                                secondaryTensor:projector_conjugate
                                                                           name:@"projector_overlaps"];
    MPSGraphTensor *weighted = [graph matrixMultiplicationWithPrimaryTensor:overlaps
                                                               secondaryTensor:d_transpose
                                                                          name:@"projector_weighting"];
    MPSGraphTensor *nonlocal_output =
        [graph matrixMultiplicationWithPrimaryTensor:weighted
                                     secondaryTensor:projector_transpose
                                                name:@"nonlocal_output"];
    CachedResidentGraph cached{
        graph, grid_input, compact_input, potential, kinetic,
        projector_conjugate, d_transpose, projector_transpose,
        local_output, nonlocal_output, grid_shape, compact_shape,
        potential_shape, projector_conjugate_shape, d_shape,
        projector_transpose_shape};
    resident_graph_cache.emplace(key, cached);
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

extern "C" void *rpd_metal_hamiltonian_create(
    const float *kinetic_interleaved,
    const float *projector_conjugate_interleaved,
    const float *d_transpose_interleaved,
    const float *projector_transpose_interleaved,
    size_t nx, size_t ny, size_t nz,
    size_t dimension, size_t projector_count, size_t max_batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        if (!kinetic_interleaved || !projector_conjugate_interleaved ||
            !d_transpose_interleaved || !projector_transpose_interleaved ||
            nx == 0 || ny == 0 || nz == 0 || dimension == 0 ||
            projector_count == 0 || max_batch == 0) {
            copy_message(error_buffer, error_capacity, @"invalid resident-Hamiltonian arguments");
            return nullptr;
        }
        id<MTLDevice> device = shared_device();
        id<MTLCommandQueue> queue = shared_queue();
        if (device == nil || queue == nil) {
            copy_message(error_buffer, error_capacity, @"failed to create Metal device or queue");
            return nullptr;
        }
        if (nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
            nx * ny * nz > SIZE_MAX / max_batch ||
            dimension > SIZE_MAX / max_batch ||
            dimension > SIZE_MAX / projector_count ||
            projector_count > SIZE_MAX / projector_count) {
            copy_message(error_buffer, error_capacity, @"resident-Hamiltonian allocation overflow");
            return nullptr;
        }
        const size_t grid_count = nx * ny * nz;
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t grid_batch_bytes = grid_count * max_batch * complex_bytes;
        const size_t compact_batch_bytes = dimension * max_batch * complex_bytes;
        const size_t grid_bytes = grid_count * complex_bytes;
        const size_t projector_bytes = dimension * projector_count * complex_bytes;
        const size_t d_bytes = projector_count * projector_count * complex_bytes;
        const uint64_t project_limit = std::min((uint64_t)(12ULL << 30),
                                                device.recommendedMaxWorkingSetSize);
        uint64_t total = 0;
        const size_t allocations[] = {
            grid_batch_bytes, compact_batch_bytes, grid_bytes, grid_bytes,
            projector_bytes, d_bytes, projector_bytes};
        for (size_t bytes : allocations) {
            if (bytes > project_limit - total) {
                copy_message(error_buffer, error_capacity,
                             @"resident Hamiltonian exceeds the 12 GiB project limit");
                return nullptr;
            }
            total += bytes;
        }
        ResidentHamiltonian *context = new ResidentHamiltonian{};
        context->device = device;
        context->queue = queue;
        context->nx = nx;
        context->ny = ny;
        context->nz = nz;
        context->grid_count = grid_count;
        context->dimension = dimension;
        context->projector_count = projector_count;
        context->max_batch = max_batch;
        context->peak_device_allocated_bytes = device.currentAllocatedSize;
        context->grid_input = [device newBufferWithLength:grid_batch_bytes
                                               options:MTLResourceStorageModeShared];
        context->compact_input = [device newBufferWithLength:compact_batch_bytes
                                                  options:MTLResourceStorageModeShared];
        context->potential = [device newBufferWithLength:grid_bytes
                                             options:MTLResourceStorageModeShared];
        context->kinetic = [device newBufferWithBytes:kinetic_interleaved
                                               length:grid_bytes
                                              options:MTLResourceStorageModeShared];
        context->projector_conjugate =
            [device newBufferWithBytes:projector_conjugate_interleaved
                                length:projector_bytes
                               options:MTLResourceStorageModeShared];
        context->d_transpose = [device newBufferWithBytes:d_transpose_interleaved
                                                    length:d_bytes
                                                   options:MTLResourceStorageModeShared];
        context->projector_transpose =
            [device newBufferWithBytes:projector_transpose_interleaved
                                length:projector_bytes
                               options:MTLResourceStorageModeShared];
        if (context->grid_input == nil || context->compact_input == nil ||
            context->potential == nil || context->kinetic == nil ||
            context->projector_conjugate == nil || context->d_transpose == nil ||
            context->projector_transpose == nil) {
            delete context;
            copy_message(error_buffer, error_capacity,
                         @"failed to allocate resident Metal buffers");
            return nullptr;
        }
        return context;
    }
}

extern "C" int rpd_metal_hamiltonian_apply(
    void *opaque_context,
    const float *grid_input_interleaved,
    const float *compact_input_interleaved,
    const float *potential_interleaved,
    float *grid_output_interleaved,
    float *nonlocal_output_interleaved,
    size_t batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !grid_input_interleaved || !compact_input_interleaved ||
            !potential_interleaved || !grid_output_interleaved ||
            !nonlocal_output_interleaved || batch == 0 || batch > context->max_batch) {
            copy_message(error_buffer, error_capacity, @"invalid resident-Hamiltonian apply arguments");
            return 2;
        }
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t grid_batch_bytes = context->grid_count * batch * complex_bytes;
        const size_t compact_batch_bytes = context->dimension * batch * complex_bytes;
        const size_t grid_bytes = context->grid_count * complex_bytes;
        std::memcpy(context->grid_input.contents, grid_input_interleaved, grid_batch_bytes);
        std::memcpy(context->compact_input.contents, compact_input_interleaved, compact_batch_bytes);
        std::memcpy(context->potential.contents, potential_interleaved, grid_bytes);

        CachedResidentGraph cached = resident_graph_for(
            context->nx, context->ny, context->nz, context->dimension,
            context->projector_count, batch);
        MPSGraphTensorData *grid_input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->grid_input shape:cached.grid_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *compact_input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->compact_input shape:cached.compact_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *potential_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->potential shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *kinetic_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->kinetic shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *projector_conjugate_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->projector_conjugate
                        shape:cached.projector_conjugate_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *d_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->d_transpose shape:cached.d_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *projector_transpose_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->projector_transpose
                        shape:cached.projector_transpose_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{
                                                                      cached.grid_input: grid_input_data,
                                                                      cached.compact_input: compact_input_data,
                                                                      cached.potential: potential_data,
                                                                      cached.kinetic: kinetic_data,
                                                                      cached.projector_conjugate: projector_conjugate_data,
                                                                      cached.d_transpose: d_data,
                                                                      cached.projector_transpose: projector_transpose_data
                                                                  }
                                                          targetTensors:@[
                                                              cached.local_output,
                                                              cached.nonlocal_output
                                                          ]
                                                       targetOperations:nil];
            MPSGraphTensorData *local_output = result[cached.local_output];
            MPSGraphTensorData *nonlocal_output = result[cached.nonlocal_output];
            [[local_output mpsndarray] readBytes:grid_output_interleaved strideBytes:nil];
            [[nonlocal_output mpsndarray] readBytes:nonlocal_output_interleaved strideBytes:nil];
            context->peak_device_allocated_bytes = std::max(
                context->peak_device_allocated_bytes,
                static_cast<uint64_t>(context->device.currentAllocatedSize));
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" void rpd_metal_hamiltonian_destroy(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    delete context;
}

extern "C" uint64_t rpd_metal_hamiltonian_peak_allocated_bytes(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    return context ? context->peak_device_allocated_bytes : 0;
}
