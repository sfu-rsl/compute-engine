#pragma once
#include "solver/gpu_buffer.hpp"
#include <solver/interfaces.h>
#include <memory>
#include <sycl/sycl.hpp>
namespace compute
{

    class SYCLComputeEngine;

    template <typename DataType>
    class SYCLComputeBuffer : GPUBufferInterface<DataType>
    {
        friend class SYCLComputeEngine;

    public:

        ~SYCLComputeBuffer() {
            if (host_alloc) {
                if (buffer_type == BufferType::DeviceCached) {
                    // std::free(host_alloc);
                    delete[] host_alloc;
                }
                else {
                    sycl::free(host_alloc, queue);
                }
            }
            if (device_alloc) {
                sycl::free(device_alloc, queue);
            }

        }

        size_t size() override
        {
            return num_elements;
        }

        size_t element_size() override
        {
            return sizeof(DataType);
        }

        size_t mem_size() override
        {
            return size() * element_size();
        }

        DataType *map() override
        {
            return host_alloc;
        }


        BufferType get_buffer_type() const override
        {
            return buffer_type;
        }

        SYCLComputeBuffer(SYCLComputeEngine *engine, sycl::queue& queue, BufferType buffer_type, DataType* data, size_t n) : engine(engine), queue(queue), buffer_type(buffer_type), num_elements(n), host_alloc(nullptr), device_alloc(nullptr)
        {
            // initialize underlying resources
            if (buffer_type == BufferType::Device || buffer_type ==  BufferType::DeviceCached || buffer_type == BufferType::Host) {
                if (buffer_type == BufferType::DeviceCached) {
                    host_alloc = new DataType[n];
                }
                else {
                    host_alloc = sycl::malloc_host<DataType>(n, queue);
                }
                if (data) {
                    std::memcpy(host_alloc, data, n*sizeof(DataType));
                }
            }

            if (buffer_type == BufferType::Device || buffer_type == BufferType::DeviceCached || buffer_type == BufferType::Storage) {
                device_alloc = sycl::malloc_device<DataType>(n, queue);
            }
        };

        // make protected?
        // Returns ptr for memory to use for sycl op. Returns device alloc when available, otherwise host alloc. 
        // This behaviour is for compatibility with the Vulkan backend - rework this later if needed.
        DataType* get_op_ptr() {
            return device_alloc ? device_alloc : host_alloc;
        }

    private:
        SYCLComputeEngine *engine;
        sycl::queue& queue;
        BufferType buffer_type;
        size_t num_elements;

        DataType* host_alloc;
        DataType* device_alloc;
    };

    template <typename DataType>
    using SCBPtr = std::shared_ptr<SYCLComputeBuffer<DataType>>;

}