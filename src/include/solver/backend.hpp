#pragma once

#include <memory>

// No default backend

namespace compute
{

    template <typename DataType>
    using BufferPtr = std::shared_ptr<GPUBuffer<DataType>>;

}