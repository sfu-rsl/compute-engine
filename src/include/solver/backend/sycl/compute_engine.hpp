#pragma once

#include <cmath>
#include <hipSYCL/sycl/device_selector.hpp>
#include <hipSYCL/sycl/info/device.hpp>
#include <hipSYCL/sycl/queue.hpp>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <sycl/sycl.hpp>
#include <solver/interfaces.h>

#include <solver/matrix_types.hpp>
#include "compute_buffer.hpp"

namespace compute
{

    class SYCLSolverSeq;

    class SYCLComputeEngine
    {
    public:
        SYCLComputeEngine(): queue(sycl::gpu_selector_v), sync_queue(sycl::gpu_selector_v, {sycl::property::queue::in_order()})
        {

            // store subgroup properties
            subgroup_size = queue.get_device().get_info<sycl::info::device::sub_group_sizes>().front();

        }

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


        uint32_t get_subgroup_size() const
        {
            return static_cast<uint32_t>(subgroup_size);
        }
        // Create object for recording linear solver sequence
        std::shared_ptr<SYCLSolverSeq> create_op_sequence()
        {
            return std::make_shared<SYCLSolverSeq>(this, sync_queue);
        }

    private:
        sycl::queue queue;
        sycl::queue sync_queue;
        size_t subgroup_size;

    // public:

        // sycl::queue& get_queue() {return queue;}

    };
}

#include "solver.hpp"