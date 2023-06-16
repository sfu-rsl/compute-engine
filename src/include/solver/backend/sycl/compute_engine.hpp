#pragma once

#include <solver/matrix_types.hpp>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sycl/sycl.hpp>
#include <solver/interfaces.h>

#include "compute_buffer.hpp"

namespace compute
{

    class SYCLSolverSeq;

    class SYCLComputeEngine
    {
    public:
        SYCLComputeEngine();

        template <typename DataType>
        SCBPtr<DataType> create_buffer(DataType *data, size_t size, BufferType buffer_type)
        {
            // create tensor and buffer wrapper
            if (size == 0)
            {
                throw std::runtime_error("Tried to create a zero-length buffer!");
            }            
            return std::make_shared<SYCLComputeBuffer<DataType>>(this, queue, buffer_type, data, size);
        }


        uint32_t get_subgroup_size() const;
        // Create object for recording linear solver sequence
        std::shared_ptr<SYCLSolverSeq> create_op_sequence();

    private:
        sycl::queue queue;
        sycl::queue sync_queue;
        size_t subgroup_size;

    };
}

#include "solver.hpp"