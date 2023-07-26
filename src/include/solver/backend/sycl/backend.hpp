#pragma once
#include <memory>
#include <vector>
#include <mutex>

#include <solver/matrix_types.hpp>
#include <solver/gpu_buffer.hpp>

#include "compute_buffer.hpp"
#include "compute_engine.hpp"

#ifndef CE_SYCL
#define CE_SYCL
#endif

namespace compute
{
    // Forward Declarations
    class SYCLComputeEngine;

    template <typename DataType>
    class SYCLComputeBuffer;

    class SYCLSolverSeq;

    // Aliases
    using ComputeEngine = SYCLComputeEngine;
    template <typename DataType>
    using GPUBuffer = SYCLComputeBuffer<DataType>;

    using SolverSeq = SYCLSolverSeq;
}