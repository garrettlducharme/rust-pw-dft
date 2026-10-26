#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
    MPSGraphTensor *basis_indices;
    MPSGraphTensor *projector_conjugate;
    MPSGraphTensor *d_transpose;
    MPSGraphTensor *projector_transpose;
    MPSGraphTensor *hamiltonian_output;
    MPSShape *grid_shape;
    MPSShape *compact_shape;
    MPSShape *potential_shape;
    MPSShape *indices_shape;
    MPSShape *projector_conjugate_shape;
    MPSShape *d_shape;
    MPSShape *projector_transpose_shape;
};

struct CachedPartialGramGraph {
    MPSGraph *graph;
    MPSGraphTensor *left;
    MPSGraphTensor *right;
    MPSGraphTensor *output;
    MPSShape *left_shape;
    MPSShape *right_shape;
};

struct CachedPartialNormGraph {
    MPSGraph *graph;
    MPSGraphTensor *input;
    MPSGraphTensor *output;
    MPSShape *input_shape;
};

struct CachedResidentRitzGraph {
    MPSGraph *graph;
    MPSGraphTensor *basis;
    MPSGraphTensor *applied_basis;
    MPSGraphTensor *coefficients;
    MPSGraphTensor *eigenvalues;
    MPSGraphTensor *ritz;
    MPSGraphTensor *applied_ritz;
    MPSGraphTensor *residual;
    MPSShape *basis_shape;
    MPSShape *coefficients_shape;
    MPSShape *eigenvalues_shape;
    MPSShape *output_shape;
};

struct CachedResidentPreconditionGraph {
    MPSGraph *graph;
    MPSGraphTensor *residual;
    MPSGraphTensor *kinetic;
    MPSGraphTensor *eigenvalues;
    MPSGraphTensor *active_mask;
    MPSGraphTensor *correction;
    MPSShape *vector_shape;
    MPSShape *kinetic_shape;
    MPSShape *root_shape;
};

struct CachedCorrectionUpdateGraph {
    MPSGraph *graph;
    MPSGraphTensor *candidate;
    MPSGraphTensor *exclusion;
    MPSGraphTensor *coefficients;
    MPSGraphTensor *output;
    MPSShape *candidate_shape;
    MPSShape *exclusion_shape;
    MPSShape *coefficient_shape;
};

struct CachedCorrectionScaleGraph {
    MPSGraph *graph;
    MPSGraphTensor *candidate;
    MPSGraphTensor *scale;
    MPSGraphTensor *output;
    MPSShape *candidate_shape;
    MPSShape *scale_shape;
};

struct ResidentHamiltonian {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> compact_scatter_pipeline;
    size_t nx;
    size_t ny;
    size_t nz;
    size_t grid_count;
    size_t dimension;
    size_t projector_count;
    size_t max_batch;
    size_t reduction_chunks;
    size_t reduction_dimension;
    size_t resident_count;
    size_t resident_ritz_count;
    size_t resident_accepted_correction_count;
    bool resident_recording;
    bool potential_initialized;
    bool resident_corrections_ready;
    bool resident_correction_work_valid;
    bool resident_correction_work_is_a;
    uint64_t peak_device_allocated_bytes;
    id<MTLBuffer> grid_input;
    id<MTLBuffer> compact_input;
    id<MTLBuffer> potential;
    id<MTLBuffer> kinetic;
    id<MTLBuffer> basis_indices;
    id<MTLBuffer> projector_conjugate;
    id<MTLBuffer> d_transpose;
    id<MTLBuffer> projector_transpose;
    id<MTLBuffer> hamiltonian_output;
    id<MTLBuffer> reduction_left;
    id<MTLBuffer> reduction_right;
    id<MTLBuffer> norm_input;
    id<MTLBuffer> ritz_coefficients;
    id<MTLBuffer> ritz_eigenvalues;
    id<MTLBuffer> resident_ritz;
    id<MTLBuffer> resident_applied_ritz;
    id<MTLBuffer> resident_residual;
    id<MTLBuffer> resident_kinetic;
    id<MTLBuffer> resident_eigenvalues_real;
    id<MTLBuffer> resident_active_mask;
    id<MTLBuffer> resident_correction;
    id<MTLBuffer> resident_correction_work_a;
    id<MTLBuffer> resident_correction_work_b;
    id<MTLBuffer> resident_accepted_corrections;
    id<MTLBuffer> resident_orthogonalization_coefficients;
};

std::mutex cache_mutex;
std::unordered_map<std::string, CachedGraph> graph_cache;
std::unordered_map<std::string, CachedLocalGraph> local_graph_cache;
std::unordered_map<std::string, CachedMatrixGraph> matrix_graph_cache;
std::unordered_map<std::string, CachedResidentGraph> resident_graph_cache;
std::unordered_map<std::string, CachedPartialGramGraph> partial_gram_graph_cache;
std::unordered_map<std::string, CachedPartialNormGraph> partial_norm_graph_cache;
std::unordered_map<std::string, CachedResidentRitzGraph> resident_ritz_graph_cache;
std::unordered_map<std::string, CachedResidentPreconditionGraph>
    resident_precondition_graph_cache;
std::unordered_map<std::string, CachedCorrectionUpdateGraph>
    correction_update_graph_cache;
std::unordered_map<std::string, CachedCorrectionScaleGraph>
    correction_scale_graph_cache;

id<MTLDevice> shared_device() {
    static id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return device;
}

id<MTLCommandQueue> shared_queue() {
    static id<MTLCommandQueue> queue = [shared_device() newCommandQueue];
    return queue;
}

id<MTLComputePipelineState> make_compact_scatter_pipeline(
    id<MTLDevice> device, NSString **error_message) {
    NSString *source = [NSString stringWithUTF8String:R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void scatter_compact_to_grid(
    device const float2 *compact [[buffer(0)]],
    device float2 *grid [[buffer(1)]],
    device const int *basis_indices [[buffer(2)]],
    constant uint &dimension [[buffer(3)]],
    constant uint &grid_count [[buffer(4)]],
    constant uint &row_stride [[buffer(5)]],
    constant uint &batch [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    const uint value_count = batch * dimension;
    if (gid >= value_count) return;
    const uint state = gid / dimension;
    const uint basis = gid - state * dimension;
    const int grid_index = basis_indices[basis];
    grid[state * grid_count + uint(grid_index)] =
        compact[state * row_stride + basis];
}
)metal"];
    NSError *library_error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:source
                                                  options:nil
                                                    error:&library_error];
    if (library == nil) {
        if (error_message) *error_message = library_error.localizedDescription;
        return nil;
    }
    id<MTLFunction> function = [library newFunctionWithName:@"scatter_compact_to_grid"];
    if (function == nil) {
        if (error_message) *error_message = @"Metal scatter function is unavailable";
        return nil;
    }
    NSError *pipeline_error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&pipeline_error];
    if (pipeline == nil && error_message) {
        *error_message = pipeline_error.localizedDescription;
    }
    return pipeline;
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
    MPSShape *indices_shape = @[@(batch), @(dimension)];
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
    MPSGraphTensor *basis_indices = [graph placeholderWithShape:indices_shape
                                                       dataType:MPSDataTypeInt32
                                                           name:@"basis_grid_indices"];
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
    MPSGraphTensor *flattened_local = [graph reshapeTensor:local_output
                                                  withShape:@[@(batch), @(nx * ny * nz)]
                                                       name:@"flattened_local_output"];
    // MPSGraph's gather kernel rejects complex tensors on current macOS.
    // Gather each FP32 component independently and reconstruct the complex
    // compact basis without leaving the graph or reading the FFT grid back.
    MPSGraphTensor *local_real = [graph realPartOfTensor:flattened_local
                                                    name:@"local_real"];
    MPSGraphTensor *local_imaginary = [graph imaginaryPartOfTensor:flattened_local
                                                              name:@"local_imaginary"];
    MPSGraphTensor *compact_real = [graph gatherAlongAxis:1
                                        withUpdatesTensor:local_real
                                            indicesTensor:basis_indices
                                                     name:@"compact_local_real"];
    MPSGraphTensor *compact_imaginary = [graph gatherAlongAxis:1
                                             withUpdatesTensor:local_imaginary
                                                 indicesTensor:basis_indices
                                                          name:@"compact_local_imaginary"];
    MPSGraphTensor *compact_local = [graph complexTensorWithRealTensor:compact_real
                                                    imaginaryTensor:compact_imaginary
                                                               name:@"compact_local_output"];
    MPSGraphTensor *hamiltonian_output = [graph additionWithPrimaryTensor:compact_local
                                                              secondaryTensor:nonlocal_output
                                                                         name:@"compact_hamiltonian_output"];
    CachedResidentGraph cached{
        graph, grid_input, compact_input, potential, kinetic, basis_indices,
        projector_conjugate, d_transpose, projector_transpose,
        hamiltonian_output, grid_shape, compact_shape,
        potential_shape, indices_shape, projector_conjugate_shape, d_shape,
        projector_transpose_shape};
    resident_graph_cache.emplace(key, cached);
    return cached;
}

CachedPartialGramGraph partial_gram_graph_for(size_t chunks,
                                              size_t left_batch,
                                              size_t right_batch) {
    constexpr size_t chunk_size = 256;
    std::string key = std::to_string(left_batch) + ":" +
                      std::to_string(right_batch) + ":" + std::to_string(chunks);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = partial_gram_graph_cache.find(key);
    if (found != partial_gram_graph_cache.end()) return found->second;

    MPSShape *left_shape = @[@(left_batch), @(chunks), @(chunk_size)];
    MPSShape *right_shape = @[@(right_batch), @(chunks), @(chunk_size)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *left_input = [graph placeholderWithShape:left_shape
                                                   dataType:MPSDataTypeComplexFloat32
                                                       name:@"overlap_left_input"];
    MPSGraphTensor *right_input = [graph placeholderWithShape:right_shape
                                                    dataType:MPSDataTypeComplexFloat32
                                                        name:@"overlap_right_input"];
    MPSGraphTensor *conjugate = [graph conjugateWithTensor:left_input
                                                      name:@"overlap_conjugate"];
    MPSGraphTensor *left = [graph reshapeTensor:conjugate
                                      withShape:@[@(left_batch), @1, @(chunks), @(chunk_size)]
                                           name:@"overlap_left"];
    MPSGraphTensor *right = [graph reshapeTensor:right_input
                                       withShape:@[@1, @(right_batch), @(chunks), @(chunk_size)]
                                            name:@"overlap_right"];
    MPSGraphTensor *products = [graph multiplicationWithPrimaryTensor:left
                                                       secondaryTensor:right
                                                                  name:@"overlap_products"];
    MPSGraphTensor *output = [graph reductionSumWithTensor:products
                                                       axis:3
                                                       name:@"overlap_partials"];
    CachedPartialGramGraph cached{
        graph, left_input, right_input, output, left_shape, right_shape};
    partial_gram_graph_cache.emplace(key, cached);
    return cached;
}

CachedPartialNormGraph partial_norm_graph_for(size_t chunks, size_t batch) {
    constexpr size_t chunk_size = 256;
    std::string key = std::to_string(batch) + ":" + std::to_string(chunks);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = partial_norm_graph_cache.find(key);
    if (found != partial_norm_graph_cache.end()) return found->second;

    MPSShape *input_shape = @[@(batch), @(chunks), @(chunk_size)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *input = [graph placeholderWithShape:input_shape
                                               dataType:MPSDataTypeComplexFloat32
                                                   name:@"residual_vectors"];
    MPSGraphTensor *conjugate = [graph conjugateWithTensor:input
                                                      name:@"residual_conjugate"];
    MPSGraphTensor *squares = [graph multiplicationWithPrimaryTensor:conjugate
                                                        secondaryTensor:input
                                                                   name:@"residual_norm_squares"];
    MPSGraphTensor *output = [graph reductionSumWithTensor:squares
                                                       axis:2
                                                       name:@"residual_norm_partials"];
    CachedPartialNormGraph cached{graph, input, output, input_shape};
    partial_norm_graph_cache.emplace(key, cached);
    return cached;
}

CachedResidentRitzGraph resident_ritz_graph_for(size_t reduction_dimension,
                                                size_t subspace,
                                                size_t roots) {
    std::string key = std::to_string(roots) + ":" + std::to_string(subspace) +
                      ":" + std::to_string(reduction_dimension);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = resident_ritz_graph_cache.find(key);
    if (found != resident_ritz_graph_cache.end()) return found->second;

    MPSShape *basis_shape = @[@(subspace), @(reduction_dimension)];
    MPSShape *coefficients_shape = @[@(roots), @(subspace)];
    MPSShape *eigenvalues_shape = @[@(roots), @1];
    MPSShape *output_shape = @[@(roots), @(reduction_dimension)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *basis = [graph placeholderWithShape:basis_shape
                                              dataType:MPSDataTypeComplexFloat32
                                                  name:@"resident_basis"];
    MPSGraphTensor *applied_basis =
        [graph placeholderWithShape:basis_shape
                           dataType:MPSDataTypeComplexFloat32
                               name:@"resident_applied_basis"];
    MPSGraphTensor *coefficients =
        [graph placeholderWithShape:coefficients_shape
                           dataType:MPSDataTypeComplexFloat32
                               name:@"ritz_coefficients"];
    MPSGraphTensor *eigenvalues =
        [graph placeholderWithShape:eigenvalues_shape
                           dataType:MPSDataTypeComplexFloat32
                               name:@"ritz_eigenvalues"];
    MPSGraphTensor *ritz = [graph matrixMultiplicationWithPrimaryTensor:coefficients
                                                        secondaryTensor:basis
                                                                   name:@"resident_ritz"];
    MPSGraphTensor *applied_ritz =
        [graph matrixMultiplicationWithPrimaryTensor:coefficients
                                     secondaryTensor:applied_basis
                                                name:@"resident_applied_ritz"];
    MPSGraphTensor *scaled_ritz = [graph multiplicationWithPrimaryTensor:ritz
                                                         secondaryTensor:eigenvalues
                                                                    name:@"scaled_resident_ritz"];
    MPSGraphTensor *residual = [graph subtractionWithPrimaryTensor:applied_ritz
                                                   secondaryTensor:scaled_ritz
                                                              name:@"resident_residual"];
    CachedResidentRitzGraph cached{
        graph, basis, applied_basis, coefficients, eigenvalues, ritz,
        applied_ritz, residual, basis_shape, coefficients_shape,
        eigenvalues_shape, output_shape};
    resident_ritz_graph_cache.emplace(key, cached);
    return cached;
}

CachedResidentPreconditionGraph resident_precondition_graph_for(
    size_t reduction_dimension, size_t roots) {
    std::string key = std::to_string(roots) + ":" +
                      std::to_string(reduction_dimension);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = resident_precondition_graph_cache.find(key);
    if (found != resident_precondition_graph_cache.end()) return found->second;

    MPSShape *vector_shape = @[@(roots), @(reduction_dimension)];
    MPSShape *kinetic_shape = @[@1, @(reduction_dimension)];
    MPSShape *root_shape = @[@(roots), @1];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *residual = [graph placeholderWithShape:vector_shape
                                                  dataType:MPSDataTypeComplexFloat32
                                                      name:@"resident_residual"];
    MPSGraphTensor *kinetic = [graph placeholderWithShape:kinetic_shape
                                                 dataType:MPSDataTypeFloat32
                                                     name:@"compact_kinetic"];
    MPSGraphTensor *eigenvalues = [graph placeholderWithShape:root_shape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"ritz_eigenvalues_real"];
    MPSGraphTensor *active_mask = [graph placeholderWithShape:root_shape
                                                     dataType:MPSDataTypeFloat32
                                                         name:@"active_root_mask"];
    MPSGraphTensor *denominator = [graph subtractionWithPrimaryTensor:kinetic
                                                      secondaryTensor:eigenvalues
                                                                 name:@"precondition_denominator"];
    MPSGraphTensor *absolute = [graph absoluteWithTensor:denominator
                                                   name:@"precondition_absolute_denominator"];
    MPSGraphTensor *floor = [graph constantWithScalar:0.25
                                             dataType:MPSDataTypeFloat32];
    MPSGraphTensor *zero = [graph constantWithScalar:0.0
                                            dataType:MPSDataTypeFloat32];
    MPSGraphTensor *negative_floor = [graph constantWithScalar:-0.25
                                                      dataType:MPSDataTypeFloat32];
    MPSGraphTensor *below_floor = [graph lessThanWithPrimaryTensor:absolute
                                                        secondaryTensor:floor
                                                                   name:@"precondition_below_floor"];
    MPSGraphTensor *is_negative = [graph lessThanWithPrimaryTensor:denominator
                                                         secondaryTensor:zero
                                                                    name:@"precondition_negative"];
    MPSGraphTensor *signed_floor = [graph selectWithPredicateTensor:is_negative
                                                       truePredicateTensor:negative_floor
                                                      falsePredicateTensor:floor
                                                                      name:@"precondition_signed_floor"];
    MPSGraphTensor *safe_denominator = [graph selectWithPredicateTensor:below_floor
                                                       truePredicateTensor:signed_floor
                                                      falsePredicateTensor:denominator
                                                                      name:@"precondition_safe_denominator"];
    MPSGraphTensor *imaginary_zero = [graph constantWithScalar:0.0
                                                         shape:vector_shape
                                                      dataType:MPSDataTypeFloat32];
    MPSGraphTensor *complex_denominator =
        [graph complexTensorWithRealTensor:safe_denominator
                           imaginaryTensor:imaginary_zero
                                      name:@"precondition_complex_denominator"];
    MPSGraphTensor *negative_residual = [graph negativeWithTensor:residual
                                                             name:@"negative_residual"];
    MPSGraphTensor *raw_correction = [graph divisionWithPrimaryTensor:negative_residual
                                                        secondaryTensor:complex_denominator
                                                                   name:@"raw_preconditioned_correction"];
    MPSGraphTensor *mask_zero = [graph constantWithScalar:0.0
                                                    shape:root_shape
                                                 dataType:MPSDataTypeFloat32];
    MPSGraphTensor *complex_mask = [graph complexTensorWithRealTensor:active_mask
                                                       imaginaryTensor:mask_zero
                                                                  name:@"complex_active_root_mask"];
    MPSGraphTensor *correction = [graph multiplicationWithPrimaryTensor:raw_correction
                                                          secondaryTensor:complex_mask
                                                                     name:@"resident_preconditioned_correction"];
    CachedResidentPreconditionGraph cached{
        graph, residual, kinetic, eigenvalues, active_mask, correction,
        vector_shape, kinetic_shape, root_shape};
    resident_precondition_graph_cache.emplace(key, cached);
    return cached;
}

CachedCorrectionUpdateGraph correction_update_graph_for(
    size_t reduction_dimension, size_t exclusion_count) {
    std::string key = std::to_string(exclusion_count) + ":" +
                      std::to_string(reduction_dimension);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = correction_update_graph_cache.find(key);
    if (found != correction_update_graph_cache.end()) return found->second;

    MPSShape *candidate_shape = @[@1, @(reduction_dimension)];
    MPSShape *exclusion_shape = @[@(exclusion_count), @(reduction_dimension)];
    MPSShape *coefficient_shape = @[@1, @(exclusion_count)];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *candidate = [graph placeholderWithShape:candidate_shape
                                                  dataType:MPSDataTypeComplexFloat32
                                                      name:@"correction_candidate"];
    MPSGraphTensor *exclusion = [graph placeholderWithShape:exclusion_shape
                                                  dataType:MPSDataTypeComplexFloat32
                                                      name:@"orthogonalization_exclusion"];
    MPSGraphTensor *coefficients = [graph placeholderWithShape:coefficient_shape
                                                     dataType:MPSDataTypeComplexFloat32
                                                         name:@"orthogonalization_coefficients"];
    MPSGraphTensor *projection = [graph matrixMultiplicationWithPrimaryTensor:coefficients
                                                              secondaryTensor:exclusion
                                                                         name:@"orthogonalization_projection"];
    MPSGraphTensor *output = [graph subtractionWithPrimaryTensor:candidate
                                                 secondaryTensor:projection
                                                            name:@"orthogonalized_correction"];
    CachedCorrectionUpdateGraph cached{
        graph, candidate, exclusion, coefficients, output,
        candidate_shape, exclusion_shape, coefficient_shape};
    correction_update_graph_cache.emplace(key, cached);
    return cached;
}

CachedCorrectionScaleGraph correction_scale_graph_for(size_t reduction_dimension) {
    std::string key = std::to_string(reduction_dimension);
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto found = correction_scale_graph_cache.find(key);
    if (found != correction_scale_graph_cache.end()) return found->second;

    MPSShape *candidate_shape = @[@1, @(reduction_dimension)];
    MPSShape *scale_shape = @[@1, @1];
    MPSGraph *graph = [[MPSGraph alloc] init];
    MPSGraphTensor *candidate = [graph placeholderWithShape:candidate_shape
                                                  dataType:MPSDataTypeComplexFloat32
                                                      name:@"correction_to_normalize"];
    MPSGraphTensor *scale = [graph placeholderWithShape:scale_shape
                                              dataType:MPSDataTypeComplexFloat32
                                                  name:@"correction_inverse_norm"];
    MPSGraphTensor *output = [graph multiplicationWithPrimaryTensor:candidate
                                                   secondaryTensor:scale
                                                              name:@"normalized_correction"];
    CachedCorrectionScaleGraph cached{
        graph, candidate, scale, output, candidate_shape, scale_shape};
    correction_scale_graph_cache.emplace(key, cached);
    return cached;
}

void invalidate_resident_corrections(ResidentHamiltonian *context) {
    context->resident_accepted_correction_count = 0;
    context->resident_corrections_ready = false;
    context->resident_correction_work_valid = false;
    context->resident_correction_work_is_a = true;
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
    const int32_t *basis_indices,
    const float *projector_conjugate_interleaved,
    const float *d_transpose_interleaved,
    const float *projector_transpose_interleaved,
    size_t nx, size_t ny, size_t nz,
    size_t dimension, size_t projector_count, size_t max_batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        if (!kinetic_interleaved || !basis_indices || !projector_conjugate_interleaved ||
            !d_transpose_interleaved || !projector_transpose_interleaved ||
            nx == 0 || ny == 0 || nz == 0 || dimension == 0 ||
            projector_count == 0 || max_batch == 0 ||
            dimension > SIZE_MAX - 255) {
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
            projector_count > SIZE_MAX / projector_count ||
            max_batch > SIZE_MAX / max_batch) {
            copy_message(error_buffer, error_capacity, @"resident-Hamiltonian allocation overflow");
            return nullptr;
        }
        const size_t grid_count = nx * ny * nz;
        if (grid_count > std::numeric_limits<uint32_t>::max() ||
            dimension > std::numeric_limits<uint32_t>::max() ||
            max_batch > std::numeric_limits<uint32_t>::max()) {
            copy_message(error_buffer, error_capacity,
                         @"resident-Hamiltonian dimensions exceed Metal uint limits");
            return nullptr;
        }
        constexpr size_t reduction_chunk_size = 256;
        const size_t reduction_chunks =
            (dimension + reduction_chunk_size - 1) / reduction_chunk_size;
        const size_t reduction_dimension = reduction_chunks * reduction_chunk_size;
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t complex_element_limit = SIZE_MAX / complex_bytes;
        if (reduction_dimension > SIZE_MAX / max_batch ||
            grid_count * max_batch > complex_element_limit ||
            dimension * max_batch > complex_element_limit ||
            grid_count > complex_element_limit ||
            dimension * projector_count > complex_element_limit ||
            projector_count * projector_count > complex_element_limit ||
            reduction_dimension * max_batch > complex_element_limit ||
            max_batch * max_batch > complex_element_limit ||
            max_batch > complex_element_limit ||
            dimension * max_batch > SIZE_MAX / sizeof(int32_t)) {
            copy_message(error_buffer, error_capacity,
                         @"resident-Hamiltonian byte-size overflow");
            return nullptr;
        }
        const size_t grid_batch_bytes = grid_count * max_batch * complex_bytes;
        const size_t compact_batch_bytes = dimension * max_batch * complex_bytes;
        const size_t grid_bytes = grid_count * complex_bytes;
        const size_t index_bytes = dimension * max_batch * sizeof(int32_t);
        const size_t projector_bytes = dimension * projector_count * complex_bytes;
        const size_t d_bytes = projector_count * projector_count * complex_bytes;
        const size_t reduction_bytes = reduction_dimension * max_batch * complex_bytes;
        const size_t coefficient_bytes = max_batch * max_batch * complex_bytes;
        const size_t eigenvalue_bytes = max_batch * complex_bytes;
        const size_t compact_kinetic_bytes = reduction_dimension * sizeof(float);
        const size_t root_scalar_bytes = max_batch * sizeof(float);
        const size_t reduction_row_bytes = reduction_dimension * complex_bytes;
        const uint64_t project_limit = std::min((uint64_t)(12ULL << 30),
                                                device.recommendedMaxWorkingSetSize);
        uint64_t total = 0;
        const size_t allocations[] = {
            grid_batch_bytes, compact_batch_bytes, grid_bytes, grid_bytes, index_bytes,
            projector_bytes, d_bytes, projector_bytes, compact_batch_bytes,
            reduction_bytes, reduction_bytes, reduction_bytes,
            coefficient_bytes, eigenvalue_bytes,
            reduction_bytes, reduction_bytes, reduction_bytes,
            compact_kinetic_bytes, root_scalar_bytes, root_scalar_bytes,
            reduction_bytes, reduction_row_bytes, reduction_row_bytes,
            reduction_bytes, eigenvalue_bytes};
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
        NSString *scatter_error = nil;
        context->compact_scatter_pipeline =
            make_compact_scatter_pipeline(device, &scatter_error);
        if (context->compact_scatter_pipeline == nil) {
            delete context;
            copy_message(error_buffer, error_capacity,
                         scatter_error ?: @"failed to create compact scatter pipeline");
            return nullptr;
        }
        context->nx = nx;
        context->ny = ny;
        context->nz = nz;
        context->grid_count = grid_count;
        context->dimension = dimension;
        context->projector_count = projector_count;
        context->max_batch = max_batch;
        context->reduction_chunks = reduction_chunks;
        context->reduction_dimension = reduction_dimension;
        context->resident_count = 0;
        context->resident_ritz_count = 0;
        context->resident_accepted_correction_count = 0;
        context->resident_recording = false;
        context->potential_initialized = false;
        context->resident_corrections_ready = false;
        context->resident_correction_work_valid = false;
        context->resident_correction_work_is_a = true;
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
        context->basis_indices = [device newBufferWithBytes:basis_indices
                                                     length:index_bytes
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
        context->hamiltonian_output =
            [device newBufferWithLength:compact_batch_bytes
                                options:MTLResourceStorageModeShared];
        context->reduction_left = [device newBufferWithLength:reduction_bytes
                                                       options:MTLResourceStorageModeShared];
        context->reduction_right = [device newBufferWithLength:reduction_bytes
                                                        options:MTLResourceStorageModeShared];
        context->norm_input = [device newBufferWithLength:reduction_bytes
                                                   options:MTLResourceStorageModeShared];
        context->ritz_coefficients = [device newBufferWithLength:coefficient_bytes
                                                          options:MTLResourceStorageModeShared];
        context->ritz_eigenvalues = [device newBufferWithLength:eigenvalue_bytes
                                                         options:MTLResourceStorageModeShared];
        context->resident_ritz = [device newBufferWithLength:reduction_bytes
                                                     options:MTLResourceStorageModeShared];
        context->resident_applied_ritz = [device newBufferWithLength:reduction_bytes
                                                             options:MTLResourceStorageModeShared];
        context->resident_residual = [device newBufferWithLength:reduction_bytes
                                                         options:MTLResourceStorageModeShared];
        context->resident_kinetic = [device newBufferWithLength:compact_kinetic_bytes
                                                        options:MTLResourceStorageModeShared];
        context->resident_eigenvalues_real =
            [device newBufferWithLength:root_scalar_bytes
                                options:MTLResourceStorageModeShared];
        context->resident_active_mask = [device newBufferWithLength:root_scalar_bytes
                                                            options:MTLResourceStorageModeShared];
        context->resident_correction = [device newBufferWithLength:reduction_bytes
                                                           options:MTLResourceStorageModeShared];
        context->resident_correction_work_a =
            [device newBufferWithLength:reduction_row_bytes
                                options:MTLResourceStorageModeShared];
        context->resident_correction_work_b =
            [device newBufferWithLength:reduction_row_bytes
                                options:MTLResourceStorageModeShared];
        context->resident_accepted_corrections =
            [device newBufferWithLength:reduction_bytes
                                options:MTLResourceStorageModeShared];
        context->resident_orthogonalization_coefficients =
            [device newBufferWithLength:eigenvalue_bytes
                                options:MTLResourceStorageModeShared];
        if (context->grid_input == nil || context->compact_input == nil ||
            context->potential == nil || context->kinetic == nil ||
            context->basis_indices == nil ||
            context->projector_conjugate == nil || context->d_transpose == nil ||
            context->projector_transpose == nil || context->hamiltonian_output == nil ||
            context->reduction_left == nil ||
            context->reduction_right == nil || context->norm_input == nil ||
            context->ritz_coefficients == nil || context->ritz_eigenvalues == nil ||
            context->resident_ritz == nil || context->resident_applied_ritz == nil ||
            context->resident_residual == nil || context->resident_kinetic == nil ||
            context->resident_eigenvalues_real == nil ||
            context->resident_active_mask == nil || context->resident_correction == nil ||
            context->resident_correction_work_a == nil ||
            context->resident_correction_work_b == nil ||
            context->resident_accepted_corrections == nil ||
            context->resident_orthogonalization_coefficients == nil) {
            delete context;
            copy_message(error_buffer, error_capacity,
                         @"failed to allocate resident Metal buffers");
            return nullptr;
        }
        float *compact_kinetic =
            static_cast<float *>(context->resident_kinetic.contents);
        std::fill(compact_kinetic, compact_kinetic + reduction_dimension, 0.0f);
        for (size_t basis = 0; basis < dimension; ++basis) {
            const int32_t grid_index = basis_indices[basis];
            if (grid_index < 0 || static_cast<size_t>(grid_index) >= grid_count) {
                delete context;
                copy_message(error_buffer, error_capacity,
                             @"resident kinetic basis index is out of range");
                return nullptr;
            }
            compact_kinetic[basis] = kinetic_interleaved[2 * grid_index];
        }
        return context;
    }
}

extern "C" int rpd_metal_hamiltonian_apply(
    void *opaque_context,
    const float *grid_input_interleaved,
    const float *compact_input_interleaved,
    const float *potential_interleaved,
    float *compact_output_interleaved,
    size_t batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !grid_input_interleaved || !compact_input_interleaved ||
            !potential_interleaved || batch == 0 || batch > context->max_batch) {
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
        context->potential_initialized = true;

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
        MPSGraphTensorData *indices_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->basis_indices shape:cached.indices_shape
                     dataType:MPSDataTypeInt32];
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
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->hamiltonian_output
                        shape:cached.compact_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.grid_input: grid_input_data,
                                               cached.compact_input: compact_input_data,
                                               cached.potential: potential_data,
                                               cached.kinetic: kinetic_data,
                                               cached.basis_indices: indices_data,
                                               cached.projector_conjugate: projector_conjugate_data,
                                               cached.d_transpose: d_data,
                                               cached.projector_transpose: projector_transpose_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{cached.hamiltonian_output: output_data}];
            if (compact_output_interleaved != nullptr) {
                std::memcpy(compact_output_interleaved,
                            context->hamiltonian_output.contents,
                            compact_batch_bytes);
            }
            if (context->resident_recording) {
                if (context->resident_count + batch > context->max_batch) {
                    // A Davidson restart replaced the old subspace before this
                    // correction block was applied. Defer a full synchronization
                    // to the next projection rather than mixing both bases.
                    context->resident_recording = false;
                    context->resident_count = 0;
                    context->resident_ritz_count = 0;
                    invalidate_resident_corrections(context);
                } else {
                    const size_t resident_row_bytes =
                        context->reduction_dimension * complex_bytes;
                    const size_t compact_row_bytes = context->dimension * complex_bytes;
                    for (size_t state = 0; state < batch; ++state) {
                        const size_t resident_state = context->resident_count + state;
                        char *basis_destination =
                            static_cast<char *>(context->reduction_left.contents) +
                            resident_state * resident_row_bytes;
                        char *applied_destination =
                            static_cast<char *>(context->reduction_right.contents) +
                            resident_state * resident_row_bytes;
                        std::memset(basis_destination, 0, resident_row_bytes);
                        std::memset(applied_destination, 0, resident_row_bytes);
                        std::memcpy(basis_destination,
                                    static_cast<const char *>(context->compact_input.contents) +
                                        state * compact_row_bytes,
                                    compact_row_bytes);
                        std::memcpy(applied_destination,
                                    static_cast<const char *>(context->hamiltonian_output.contents) +
                                        state * compact_row_bytes,
                                    compact_row_bytes);
                    }
                    context->resident_count += batch;
                    context->resident_ritz_count = 0;
                    invalidate_resident_corrections(context);
                }
            }
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

extern "C" int rpd_metal_hamiltonian_partial_gram(
    void *opaque_context,
    const float *left_input_interleaved,
    const float *right_input_interleaved,
    float *partial_output_interleaved,
    size_t left_batch,
    size_t right_batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !left_input_interleaved || !right_input_interleaved ||
            !partial_output_interleaved || left_batch == 0 || right_batch == 0 ||
            left_batch > context->max_batch || right_batch > context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"invalid partial-Gram arguments");
            return 2;
        }
        context->resident_recording = false;
        context->resident_count = 0;
        context->resident_ritz_count = 0;
        invalidate_resident_corrections(context);
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t reduction_row_bytes = context->reduction_dimension * complex_bytes;
        const size_t compact_row_bytes = context->dimension * complex_bytes;
        std::memset(context->reduction_left.contents, 0,
                    context->reduction_dimension * left_batch * complex_bytes);
        std::memset(context->reduction_right.contents, 0,
                    context->reduction_dimension * right_batch * complex_bytes);
        for (size_t state = 0; state < left_batch; ++state) {
            std::memcpy(static_cast<char *>(context->reduction_left.contents) +
                            state * reduction_row_bytes,
                        left_input_interleaved + state * context->dimension * 2,
                        compact_row_bytes);
        }
        for (size_t state = 0; state < right_batch; ++state) {
            std::memcpy(static_cast<char *>(context->reduction_right.contents) +
                            state * reduction_row_bytes,
                        right_input_interleaved + state * context->dimension * 2,
                        compact_row_bytes);
        }
        CachedPartialGramGraph cached = partial_gram_graph_for(
            context->reduction_chunks, left_batch, right_batch);
        MPSGraphTensorData *left_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_left
                        shape:cached.left_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *right_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_right
                        shape:cached.right_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{
                                                                      cached.left: left_data,
                                                                      cached.right: right_data
                                                                  }
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
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

extern "C" int rpd_metal_hamiltonian_resident_begin(
    void *opaque_context,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context) {
        copy_message(error_buffer, error_capacity,
                     @"invalid resident Davidson context");
        return 2;
    }
    context->resident_count = 0;
    context->resident_ritz_count = 0;
    invalidate_resident_corrections(context);
    context->resident_recording = true;
    return 0;
}

extern "C" size_t rpd_metal_hamiltonian_resident_count(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    return context ? context->resident_count : 0;
}

extern "C" int rpd_metal_hamiltonian_resident_sync(
    void *opaque_context,
    const float *basis_interleaved,
    const float *applied_interleaved,
    size_t batch,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !basis_interleaved || !applied_interleaved || batch == 0 ||
        batch > context->max_batch) {
        copy_message(error_buffer, error_capacity,
                     @"invalid resident Davidson synchronization arguments");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    for (size_t state = 0; state < batch; ++state) {
        char *basis_destination =
            static_cast<char *>(context->reduction_left.contents) +
            state * resident_row_bytes;
        char *applied_destination =
            static_cast<char *>(context->reduction_right.contents) +
            state * resident_row_bytes;
        std::memset(basis_destination, 0, resident_row_bytes);
        std::memset(applied_destination, 0, resident_row_bytes);
        std::memcpy(basis_destination,
                    basis_interleaved + state * context->dimension * 2,
                    compact_row_bytes);
        std::memcpy(applied_destination,
                    applied_interleaved + state * context->dimension * 2,
                    compact_row_bytes);
    }
    context->resident_count = batch;
    context->resident_ritz_count = 0;
    invalidate_resident_corrections(context);
    context->resident_recording = true;
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_partial_gram(
    void *opaque_context,
    float *partial_output_interleaved,
    size_t subspace,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !partial_output_interleaved || subspace == 0 ||
            subspace != context->resident_count) {
            copy_message(error_buffer, error_capacity,
                         @"resident Davidson projection is out of sync");
            return 2;
        }
        CachedPartialGramGraph cached = partial_gram_graph_for(
            context->reduction_chunks, subspace, subspace);
        MPSGraphTensorData *left_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_left
                        shape:cached.left_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *right_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_right
                        shape:cached.right_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{
                                                                      cached.left: left_data,
                                                                      cached.right: right_data
                                                                  }
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
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

extern "C" int rpd_metal_hamiltonian_resident_form_ritz_residuals(
    void *opaque_context,
    const float *coefficients_interleaved,
    const float *eigenvalues_interleaved,
    size_t subspace,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !coefficients_interleaved || !eigenvalues_interleaved ||
            subspace == 0 || roots == 0 || subspace != context->resident_count ||
            subspace > context->max_batch || roots > context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"resident Ritz construction is out of sync");
            return 2;
        }
        const size_t complex_bytes = 2 * sizeof(float);
        std::memcpy(context->ritz_coefficients.contents, coefficients_interleaved,
                    roots * subspace * complex_bytes);
        std::memcpy(context->ritz_eigenvalues.contents, eigenvalues_interleaved,
                    roots * complex_bytes);
        invalidate_resident_corrections(context);
        float *real_eigenvalues =
            static_cast<float *>(context->resident_eigenvalues_real.contents);
        for (size_t root = 0; root < roots; ++root) {
            real_eigenvalues[root] = eigenvalues_interleaved[2 * root];
        }

        CachedResidentRitzGraph cached = resident_ritz_graph_for(
            context->reduction_dimension, subspace, roots);
        MPSGraphTensorData *basis_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_left
                        shape:cached.basis_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *applied_basis_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->reduction_right
                        shape:cached.basis_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *coefficient_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->ritz_coefficients
                        shape:cached.coefficients_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *eigenvalue_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->ritz_eigenvalues
                        shape:cached.eigenvalues_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *ritz_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_ritz
                        shape:cached.output_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *applied_ritz_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_applied_ritz
                        shape:cached.output_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *residual_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_residual
                        shape:cached.output_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.basis: basis_data,
                                               cached.applied_basis: applied_basis_data,
                                               cached.coefficients: coefficient_data,
                                               cached.eigenvalues: eigenvalue_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{
                                   cached.ritz: ritz_data,
                                   cached.applied_ritz: applied_ritz_data,
                                   cached.residual: residual_data
                               }];
            context->resident_ritz_count = roots;
            context->peak_device_allocated_bytes = std::max(
                context->peak_device_allocated_bytes,
                static_cast<uint64_t>(context->device.currentAllocatedSize));
        } @catch (NSException *exception) {
            context->resident_ritz_count = 0;
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_precondition(
    void *opaque_context,
    const float *active_mask,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !active_mask || roots == 0 ||
            roots != context->resident_ritz_count) {
            copy_message(error_buffer, error_capacity,
                         @"resident preconditioner is unavailable or out of sync");
            return 2;
        }
        for (size_t root = 0; root < roots; ++root) {
            if (active_mask[root] != 0.0f && active_mask[root] != 1.0f) {
                copy_message(error_buffer, error_capacity,
                             @"resident preconditioner mask must contain only zero or one");
                return 2;
            }
        }
        std::memcpy(context->resident_active_mask.contents, active_mask,
                    roots * sizeof(float));

        CachedResidentPreconditionGraph cached = resident_precondition_graph_for(
            context->reduction_dimension, roots);
        MPSGraphTensorData *residual_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_residual
                        shape:cached.vector_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *kinetic_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_kinetic
                        shape:cached.kinetic_shape
                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *eigenvalue_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_eigenvalues_real
                        shape:cached.root_shape
                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *mask_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_active_mask
                        shape:cached.root_shape
                     dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *correction_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_correction
                        shape:cached.vector_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.residual: residual_data,
                                               cached.kinetic: kinetic_data,
                                               cached.eigenvalues: eigenvalue_data,
                                               cached.active_mask: mask_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{
                                   cached.correction: correction_data
                               }];
            context->resident_accepted_correction_count = 0;
            context->resident_corrections_ready = true;
            context->resident_correction_work_valid = false;
            context->resident_correction_work_is_a = true;
            context->peak_device_allocated_bytes = std::max(
                context->peak_device_allocated_bytes,
                static_cast<uint64_t>(context->device.currentAllocatedSize));
        } @catch (NSException *exception) {
            invalidate_resident_corrections(context);
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_read_corrections(
    void *opaque_context,
    float *correction_output_interleaved,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !correction_output_interleaved || roots == 0 ||
        roots != context->resident_ritz_count ||
        !context->resident_corrections_ready) {
        copy_message(error_buffer, error_capacity,
                     @"resident corrections are unavailable or out of sync");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    for (size_t root = 0; root < roots; ++root) {
        std::memcpy(correction_output_interleaved + root * context->dimension * 2,
                    static_cast<const char *>(context->resident_correction.contents) +
                        root * resident_row_bytes,
                    compact_row_bytes);
    }
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_orthogonalization_begin(
    void *opaque_context,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || roots == 0 || roots != context->resident_ritz_count ||
        !context->resident_corrections_ready) {
        copy_message(error_buffer, error_capacity,
                     @"resident orthogonalization is unavailable or out of sync");
        return 2;
    }
    context->resident_accepted_correction_count = 0;
    context->resident_correction_work_valid = false;
    context->resident_correction_work_is_a = true;
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_correction_begin(
    void *opaque_context,
    size_t root,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || root >= context->resident_ritz_count ||
            !context->resident_corrections_ready) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction root is unavailable or out of sync");
            return 2;
        }
        const size_t row_bytes =
            context->reduction_dimension * 2 * sizeof(float);
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create resident correction copy command");
            return 3;
        }
        [encoder copyFromBuffer:context->resident_correction
                   sourceOffset:root * row_bytes
                       toBuffer:context->resident_correction_work_a
              destinationOffset:0
                           size:row_bytes];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            copy_message(error_buffer, error_capacity,
                         command_buffer.error.localizedDescription);
            return 3;
        }
        context->resident_correction_work_valid = true;
        context->resident_correction_work_is_a = true;
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_correction_partial_gram(
    void *opaque_context,
    int against_accepted,
    float *partial_output_interleaved,
    size_t exclusion_count,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        const size_t expected_count = against_accepted
            ? (context ? context->resident_accepted_correction_count : 0)
            : (context ? context->resident_count : 0);
        if (!context || !partial_output_interleaved ||
            !context->resident_correction_work_valid || exclusion_count == 0 ||
            exclusion_count != expected_count ||
            (against_accepted != 0 && against_accepted != 1)) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction projection is out of sync");
            return 2;
        }
        id<MTLBuffer> exclusion = against_accepted
            ? context->resident_accepted_corrections
            : context->reduction_left;
        id<MTLBuffer> candidate = context->resident_correction_work_is_a
            ? context->resident_correction_work_a
            : context->resident_correction_work_b;
        CachedPartialGramGraph cached = partial_gram_graph_for(
            context->reduction_chunks, exclusion_count, 1);
        MPSGraphTensorData *left_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:exclusion
                        shape:cached.left_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *right_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:candidate
                        shape:cached.right_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{
                                                                      cached.left: left_data,
                                                                      cached.right: right_data
                                                                  }
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
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

extern "C" int rpd_metal_hamiltonian_resident_correction_subtract(
    void *opaque_context,
    int against_accepted,
    const float *coefficients_interleaved,
    size_t exclusion_count,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        const size_t expected_count = against_accepted
            ? (context ? context->resident_accepted_correction_count : 0)
            : (context ? context->resident_count : 0);
        if (!context || !coefficients_interleaved ||
            !context->resident_correction_work_valid || exclusion_count == 0 ||
            exclusion_count != expected_count ||
            (against_accepted != 0 && against_accepted != 1)) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction update is out of sync");
            return 2;
        }
        const size_t complex_bytes = 2 * sizeof(float);
        std::memcpy(context->resident_orthogonalization_coefficients.contents,
                    coefficients_interleaved,
                    exclusion_count * complex_bytes);
        id<MTLBuffer> exclusion = against_accepted
            ? context->resident_accepted_corrections
            : context->reduction_left;
        id<MTLBuffer> candidate = context->resident_correction_work_is_a
            ? context->resident_correction_work_a
            : context->resident_correction_work_b;
        id<MTLBuffer> output_buffer = context->resident_correction_work_is_a
            ? context->resident_correction_work_b
            : context->resident_correction_work_a;
        CachedCorrectionUpdateGraph cached = correction_update_graph_for(
            context->reduction_dimension, exclusion_count);
        MPSGraphTensorData *candidate_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:candidate
                        shape:cached.candidate_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *exclusion_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:exclusion
                        shape:cached.exclusion_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *coefficient_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_orthogonalization_coefficients
                        shape:cached.coefficient_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output_buffer
                        shape:cached.candidate_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.candidate: candidate_data,
                                               cached.exclusion: exclusion_data,
                                               cached.coefficients: coefficient_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{cached.output: output_data}];
            context->resident_correction_work_is_a =
                !context->resident_correction_work_is_a;
            context->peak_device_allocated_bytes = std::max(
                context->peak_device_allocated_bytes,
                static_cast<uint64_t>(context->device.currentAllocatedSize));
        } @catch (NSException *exception) {
            context->resident_correction_work_valid = false;
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_correction_partial_norm(
    void *opaque_context,
    float *partial_output_interleaved,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !partial_output_interleaved ||
            !context->resident_correction_work_valid) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction norm is unavailable");
            return 2;
        }
        id<MTLBuffer> candidate = context->resident_correction_work_is_a
            ? context->resident_correction_work_a
            : context->resident_correction_work_b;
        CachedPartialNormGraph cached =
            partial_norm_graph_for(context->reduction_chunks, 1);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:candidate
                        shape:cached.input_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{cached.input: input_data}
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_correction_normalize(
    void *opaque_context,
    float inverse_norm,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !context->resident_correction_work_valid ||
            !std::isfinite(inverse_norm) || inverse_norm <= 0.0f) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction normalization is invalid");
            return 2;
        }
        float scale_values[2] = {inverse_norm, 0.0f};
        std::memcpy(context->resident_orthogonalization_coefficients.contents,
                    scale_values, sizeof(scale_values));
        id<MTLBuffer> candidate = context->resident_correction_work_is_a
            ? context->resident_correction_work_a
            : context->resident_correction_work_b;
        id<MTLBuffer> output_buffer = context->resident_correction_work_is_a
            ? context->resident_correction_work_b
            : context->resident_correction_work_a;
        CachedCorrectionScaleGraph cached =
            correction_scale_graph_for(context->reduction_dimension);
        MPSGraphTensorData *candidate_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:candidate
                        shape:cached.candidate_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *scale_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_orthogonalization_coefficients
                        shape:cached.scale_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:output_buffer
                        shape:cached.candidate_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.candidate: candidate_data,
                                               cached.scale: scale_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{cached.output: output_data}];
            context->resident_correction_work_is_a =
                !context->resident_correction_work_is_a;
        } @catch (NSException *exception) {
            context->resident_correction_work_valid = false;
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_correction_accept(
    void *opaque_context,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !context->resident_correction_work_valid ||
            context->resident_accepted_correction_count >= context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction cannot be accepted");
            return 2;
        }
        id<MTLBuffer> candidate = context->resident_correction_work_is_a
            ? context->resident_correction_work_a
            : context->resident_correction_work_b;
        const size_t row_bytes =
            context->reduction_dimension * 2 * sizeof(float);
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create accepted-correction copy command");
            return 3;
        }
        [encoder copyFromBuffer:candidate
                   sourceOffset:0
                       toBuffer:context->resident_accepted_corrections
              destinationOffset:context->resident_accepted_correction_count * row_bytes
                           size:row_bytes];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            copy_message(error_buffer, error_capacity,
                         command_buffer.error.localizedDescription);
            return 3;
        }
        context->resident_accepted_correction_count += 1;
        context->resident_correction_work_valid = false;
        return 0;
    }
}

extern "C" size_t rpd_metal_hamiltonian_resident_accepted_correction_count(
    void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    return context ? context->resident_accepted_correction_count : 0;
}

extern "C" int rpd_metal_hamiltonian_resident_restart_from_ritz(
    void *opaque_context,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || roots == 0 || roots != context->resident_ritz_count ||
            roots > context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"resident Ritz restart is unavailable or out of sync");
            return 2;
        }
        const size_t row_bytes =
            context->reduction_dimension * 2 * sizeof(float);
        const size_t journal_bytes = context->max_batch * row_bytes;
        id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [command_buffer blitCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create resident Ritz restart command");
            return 3;
        }
        [encoder fillBuffer:context->reduction_left
                      range:NSMakeRange(0, journal_bytes)
                      value:0];
        [encoder fillBuffer:context->reduction_right
                      range:NSMakeRange(0, journal_bytes)
                      value:0];
        [encoder copyFromBuffer:context->resident_ritz
                   sourceOffset:0
                       toBuffer:context->reduction_left
              destinationOffset:0
                           size:roots * row_bytes];
        [encoder copyFromBuffer:context->resident_applied_ritz
                   sourceOffset:0
                       toBuffer:context->reduction_right
              destinationOffset:0
                           size:roots * row_bytes];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];
        if (command_buffer.status == MTLCommandBufferStatusError) {
            copy_message(error_buffer, error_capacity,
                         command_buffer.error.localizedDescription);
            return 3;
        }
        context->resident_count = roots;
        context->resident_recording = true;
        context->resident_accepted_correction_count = 0;
        context->resident_correction_work_valid = false;
        context->resident_correction_work_is_a = true;
        context->peak_device_allocated_bytes = std::max(
            context->peak_device_allocated_bytes,
            static_cast<uint64_t>(context->device.currentAllocatedSize));
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_apply_accepted_corrections(
    void *opaque_context,
    size_t accepted_count,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || accepted_count == 0 ||
            accepted_count != context->resident_accepted_correction_count ||
            !context->resident_corrections_ready ||
            !context->potential_initialized ||
            context->resident_count + accepted_count > context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"resident correction Hamiltonian application is out of sync");
            return 2;
        }
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t grid_batch_bytes =
            context->grid_count * accepted_count * complex_bytes;
        const size_t resident_row_bytes =
            context->reduction_dimension * complex_bytes;
        const size_t compact_row_bytes = context->dimension * complex_bytes;

        id<MTLCommandBuffer> scatter_command = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> clear_encoder =
            [scatter_command blitCommandEncoder];
        if (scatter_command == nil || clear_encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create resident scatter command");
            return 3;
        }
        [clear_encoder fillBuffer:context->grid_input
                            range:NSMakeRange(0, grid_batch_bytes)
                            value:0];
        [clear_encoder endEncoding];
        id<MTLComputeCommandEncoder> scatter_encoder =
            [scatter_command computeCommandEncoder];
        if (scatter_encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create resident scatter encoder");
            return 3;
        }
        const uint32_t dimension = static_cast<uint32_t>(context->dimension);
        const uint32_t grid_count = static_cast<uint32_t>(context->grid_count);
        const uint32_t row_stride =
            static_cast<uint32_t>(context->reduction_dimension);
        const uint32_t batch = static_cast<uint32_t>(accepted_count);
        [scatter_encoder setComputePipelineState:context->compact_scatter_pipeline];
        [scatter_encoder setBuffer:context->resident_accepted_corrections
                            offset:0 atIndex:0];
        [scatter_encoder setBuffer:context->grid_input offset:0 atIndex:1];
        [scatter_encoder setBuffer:context->basis_indices offset:0 atIndex:2];
        [scatter_encoder setBytes:&dimension length:sizeof(dimension) atIndex:3];
        [scatter_encoder setBytes:&grid_count length:sizeof(grid_count) atIndex:4];
        [scatter_encoder setBytes:&row_stride length:sizeof(row_stride) atIndex:5];
        [scatter_encoder setBytes:&batch length:sizeof(batch) atIndex:6];
        const NSUInteger value_count = accepted_count * context->dimension;
        const NSUInteger thread_count = std::min(
            value_count, context->compact_scatter_pipeline.maxTotalThreadsPerThreadgroup);
        [scatter_encoder dispatchThreads:MTLSizeMake(value_count, 1, 1)
                   threadsPerThreadgroup:MTLSizeMake(thread_count, 1, 1)];
        [scatter_encoder endEncoding];
        [scatter_command commit];
        [scatter_command waitUntilCompleted];
        if (scatter_command.status == MTLCommandBufferStatusError) {
            copy_message(error_buffer, error_capacity,
                         scatter_command.error.localizedDescription);
            return 3;
        }

        CachedResidentGraph cached = resident_graph_for(
            context->nx, context->ny, context->nz, context->dimension,
            context->projector_count, accepted_count);
        MPSGraphTensorData *grid_input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->grid_input shape:cached.grid_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *compact_input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_accepted_corrections
                        shape:cached.compact_shape
                     dataType:MPSDataTypeComplexFloat32
                     rowBytes:resident_row_bytes];
        MPSGraphTensorData *potential_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->potential shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *kinetic_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->kinetic shape:cached.potential_shape
                     dataType:MPSDataTypeComplexFloat32];
        MPSGraphTensorData *indices_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->basis_indices shape:cached.indices_shape
                     dataType:MPSDataTypeInt32];
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
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->hamiltonian_output
                        shape:cached.compact_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            [cached.graph runWithMTLCommandQueue:context->queue
                                           feeds:@{
                                               cached.grid_input: grid_input_data,
                                               cached.compact_input: compact_input_data,
                                               cached.potential: potential_data,
                                               cached.kinetic: kinetic_data,
                                               cached.basis_indices: indices_data,
                                               cached.projector_conjugate: projector_conjugate_data,
                                               cached.d_transpose: d_data,
                                               cached.projector_transpose: projector_transpose_data
                                           }
                                targetOperations:nil
                               resultsDictionary:@{cached.hamiltonian_output: output_data}];
        } @catch (NSException *exception) {
            copy_message(error_buffer, error_capacity, exception.reason);
            return 3;
        }

        id<MTLCommandBuffer> append_command = [context->queue commandBuffer];
        id<MTLBlitCommandEncoder> append_encoder =
            [append_command blitCommandEncoder];
        if (append_command == nil || append_encoder == nil) {
            copy_message(error_buffer, error_capacity,
                         @"failed to create resident append command");
            return 3;
        }
        [append_encoder fillBuffer:context->reduction_right
                             range:NSMakeRange(
                                 context->resident_count * resident_row_bytes,
                                 accepted_count * resident_row_bytes)
                             value:0];
        for (size_t correction = 0; correction < accepted_count; ++correction) {
            const size_t destination = context->resident_count + correction;
            [append_encoder copyFromBuffer:context->resident_accepted_corrections
                              sourceOffset:correction * resident_row_bytes
                                  toBuffer:context->reduction_left
                         destinationOffset:destination * resident_row_bytes
                                      size:resident_row_bytes];
            [append_encoder copyFromBuffer:context->hamiltonian_output
                              sourceOffset:correction * compact_row_bytes
                                  toBuffer:context->reduction_right
                         destinationOffset:destination * resident_row_bytes
                                      size:compact_row_bytes];
        }
        [append_encoder endEncoding];
        [append_command commit];
        [append_command waitUntilCompleted];
        if (append_command.status == MTLCommandBufferStatusError) {
            copy_message(error_buffer, error_capacity,
                         append_command.error.localizedDescription);
            return 3;
        }
        context->resident_count += accepted_count;
        context->resident_ritz_count = 0;
        context->resident_recording = true;
        invalidate_resident_corrections(context);
        context->peak_device_allocated_bytes = std::max(
            context->peak_device_allocated_bytes,
            static_cast<uint64_t>(context->device.currentAllocatedSize));
        return 0;
    }
}

extern "C" int rpd_metal_hamiltonian_resident_read_journal_tail(
    void *opaque_context,
    float *basis_output_interleaved,
    float *applied_output_interleaved,
    size_t tail_count,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !basis_output_interleaved || !applied_output_interleaved ||
        tail_count == 0 || tail_count > context->resident_count) {
        copy_message(error_buffer, error_capacity,
                     @"resident journal tail is unavailable");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    const size_t first = context->resident_count - tail_count;
    for (size_t row = 0; row < tail_count; ++row) {
        const size_t resident_row = first + row;
        std::memcpy(basis_output_interleaved + row * context->dimension * 2,
                    static_cast<const char *>(context->reduction_left.contents) +
                        resident_row * resident_row_bytes,
                    compact_row_bytes);
        std::memcpy(applied_output_interleaved + row * context->dimension * 2,
                    static_cast<const char *>(context->reduction_right.contents) +
                        resident_row * resident_row_bytes,
                    compact_row_bytes);
    }
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_read_accepted_corrections(
    void *opaque_context,
    float *correction_output_interleaved,
    size_t accepted_count,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !correction_output_interleaved || accepted_count == 0 ||
        accepted_count != context->resident_accepted_correction_count) {
        copy_message(error_buffer, error_capacity,
                     @"resident accepted corrections are unavailable or out of sync");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    for (size_t correction = 0; correction < accepted_count; ++correction) {
        std::memcpy(correction_output_interleaved + correction * context->dimension * 2,
                    static_cast<const char *>(context->resident_accepted_corrections.contents) +
                        correction * resident_row_bytes,
                    compact_row_bytes);
    }
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_read_ritz_residuals(
    void *opaque_context,
    float *ritz_output_interleaved,
    float *applied_ritz_output_interleaved,
    float *residual_output_interleaved,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !ritz_output_interleaved || !applied_ritz_output_interleaved ||
        !residual_output_interleaved || roots == 0 ||
        roots != context->resident_ritz_count) {
        copy_message(error_buffer, error_capacity,
                     @"resident Ritz vectors are unavailable or out of sync");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    for (size_t root = 0; root < roots; ++root) {
        std::memcpy(ritz_output_interleaved + root * context->dimension * 2,
                    static_cast<const char *>(context->resident_ritz.contents) +
                        root * resident_row_bytes,
                    compact_row_bytes);
        std::memcpy(applied_ritz_output_interleaved + root * context->dimension * 2,
                    static_cast<const char *>(context->resident_applied_ritz.contents) +
                        root * resident_row_bytes,
                    compact_row_bytes);
        std::memcpy(residual_output_interleaved + root * context->dimension * 2,
                    static_cast<const char *>(context->resident_residual.contents) +
                        root * resident_row_bytes,
                    compact_row_bytes);
    }
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_read_ritz(
    void *opaque_context,
    float *ritz_output_interleaved,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    if (!context || !ritz_output_interleaved || roots == 0 ||
        roots != context->resident_ritz_count) {
        copy_message(error_buffer, error_capacity,
                     @"resident Ritz vectors are unavailable or out of sync");
        return 2;
    }
    const size_t complex_bytes = 2 * sizeof(float);
    const size_t resident_row_bytes = context->reduction_dimension * complex_bytes;
    const size_t compact_row_bytes = context->dimension * complex_bytes;
    for (size_t root = 0; root < roots; ++root) {
        std::memcpy(ritz_output_interleaved + root * context->dimension * 2,
                    static_cast<const char *>(context->resident_ritz.contents) +
                        root * resident_row_bytes,
                    compact_row_bytes);
    }
    return 0;
}

extern "C" int rpd_metal_hamiltonian_resident_partial_residual_norms(
    void *opaque_context,
    float *partial_output_interleaved,
    size_t roots,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !partial_output_interleaved || roots == 0 ||
            roots != context->resident_ritz_count) {
            copy_message(error_buffer, error_capacity,
                         @"resident residual norms are unavailable or out of sync");
            return 2;
        }
        CachedPartialNormGraph cached =
            partial_norm_graph_for(context->reduction_chunks, roots);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->resident_residual
                        shape:cached.input_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{cached.input: input_data}
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
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

extern "C" int rpd_metal_hamiltonian_partial_norms(
    void *opaque_context,
    const float *input_interleaved,
    float *partial_output_interleaved,
    size_t batch,
    char *error_buffer, size_t error_capacity) {
    @autoreleasepool {
        ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
        if (!context || !input_interleaved || !partial_output_interleaved ||
            batch == 0 || batch > context->max_batch) {
            copy_message(error_buffer, error_capacity,
                         @"invalid partial-norm arguments");
            return 2;
        }
        const size_t complex_bytes = 2 * sizeof(float);
        const size_t reduction_row_bytes = context->reduction_dimension * complex_bytes;
        const size_t compact_row_bytes = context->dimension * complex_bytes;
        std::memset(context->norm_input.contents, 0,
                    context->reduction_dimension * batch * complex_bytes);
        for (size_t state = 0; state < batch; ++state) {
            std::memcpy(static_cast<char *>(context->norm_input.contents) +
                            state * reduction_row_bytes,
                        input_interleaved + state * context->dimension * 2,
                        compact_row_bytes);
        }
        CachedPartialNormGraph cached =
            partial_norm_graph_for(context->reduction_chunks, batch);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:context->norm_input
                        shape:cached.input_shape
                     dataType:MPSDataTypeComplexFloat32];
        @try {
            NSDictionary *result = [cached.graph runWithMTLCommandQueue:context->queue
                                                                  feeds:@{cached.input: input_data}
                                                          targetTensors:@[cached.output]
                                                       targetOperations:nil];
            MPSGraphTensorData *output = result[cached.output];
            [[output mpsndarray] readBytes:partial_output_interleaved strideBytes:nil];
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

extern "C" size_t rpd_metal_hamiltonian_reduction_chunks(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    return context ? context->reduction_chunks : 0;
}

extern "C" void rpd_metal_hamiltonian_destroy(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    delete context;
}

extern "C" uint64_t rpd_metal_hamiltonian_peak_allocated_bytes(void *opaque_context) {
    ResidentHamiltonian *context = static_cast<ResidentHamiltonian *>(opaque_context);
    return context ? context->peak_device_allocated_bytes : 0;
}
