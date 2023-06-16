#include <solver/backend/sycl/compute_engine.hpp>
namespace compute {

    SYCLComputeEngine::SYCLComputeEngine(): queue(sycl::gpu_selector_v), sync_queue(sycl::gpu_selector_v, {sycl::property::queue::in_order()})
    {
        // store subgroup properties
        subgroup_size = queue.get_device().get_info<sycl::info::device::sub_group_sizes>().front();

    }   

    uint32_t SYCLComputeEngine::get_subgroup_size() const
    {
        return static_cast<uint32_t>(subgroup_size);
    }
    // Create object for recording linear solver sequence
    std::shared_ptr<SYCLSolverSeq> SYCLComputeEngine::create_op_sequence()
    {
        return std::make_shared<SYCLSolverSeq>(this, sync_queue);
    }

}