cmake_minimum_required(VERSION 3.4.1)
project(spearhead VERSION 0.1.0)

execute_process(COMMAND "${CMAKE_CXX_COMPILER}" --opensycl-version RESULT_VARIABLE retvar OUTPUT_QUIET ERROR_QUIET)

if(retvar EQUAL "0")
  message("COMPILER IS SYCLCC")
  set(CE_SYCL ON)
  set(CE_VULKAN OFF)
else()
  message("COMPILER IS NOT SYCLCC")
  set(CE_SYCL OFF)
  set(CE_VULKAN ON)
endif()


if(CE_SYCL)
  set(CMAKE_CXX_STANDARD 17)
  add_compile_definitions(CE_SYCL_BACKEND=1)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)

FetchContent_Declare(
  googlebench
  URL https://github.com/google/benchmark/archive/refs/tags/v1.6.1.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebench)

include(GoogleTest)



# General
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries." FORCE)

# Kompute
if(CE_VULKAN)
  add_compile_definitions(SPDLOG_ACTIVE_LEVEL=5)
  add_compile_definitions(KOMPUTE_VK_API_MAJOR_VERSION=1 KOMPUTE_VK_API_MINOR_VERSION=2)
  set(KOMPUTE_OPT_LOG_LEVEL "Off" CACHE STRING "Kompute logging level" FORCE)
  set(KOMPUTE_OPT_DISABLE_VULKAN_VERSION_CHECK ON)
  set(KOMPUTE_OPT_RELAX ON)
  set(KOMPUTE_OPT_USE_BUILT_IN_GOOGLE_TEST OFF)
  set(KOMPUTE_OPT_DISABLE_VK_DEBUG_LAYERS ON)
  # set(KOMPUTE_OPT_USE_SPDLOG ON)
  add_subdirectory(external/kompute)
endif()


# It is necessary to pass the DEBUG or RELEASE flag accordingly to Kompute
if(CE_VULKAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1 ${KOMPUTE_EXTRA_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE=1 ${KOMPUTE_EXTRA_CXX_FLAGS} -march=native")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG=1 -Wall -Wextra")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DRELEASE=1 -march=native")
endif()


find_package(Eigen3 3.1.0 REQUIRED)
find_package(Threads REQUIRED)


if(CE_VULKAN)
add_library(compute SHARED
src/backend/vulkan/compute_engine.cpp
src/CustomOp.cpp
)
else()
add_library(compute SHARED
src/backend/sycl/compute_engine.cpp
)
endif()



# compile shaders
if(CE_VULKAN)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  file(GLOB_RECURSE SHADERS
  "shaders/*.comp"
  )

  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ 
  DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/data/)

  add_custom_target(compile_shaders
  COMMAND mkdir -p ${CMAKE_CURRENT_SOURCE_DIR}/src/include/solver/shaders
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/external/kompute/scripts/convert_shaders_custom.py -p shaders -s glslangValidator -c ${CMAKE_CURRENT_SOURCE_DIR}/src/include/solver/shaders/
  DEPENDS ${SHADERS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  add_dependencies(compute
    compile_shaders
  )

endif()


target_link_libraries(compute Threads::Threads)

if(CE_VULKAN)
  target_link_libraries(compute kompute::kompute)
  target_include_directories(compute PUBLIC external/kompute/src/include)
endif()

target_include_directories(compute 
    PUBLIC ${EIGEN3_INCLUDE_DIR}
    PUBLIC src/include
)


# Benchmarks
add_executable(sbm_bench
  src/sbm_bench.cpp
)



target_link_libraries(sbm_bench
    # kompute::kompute
    benchmark::benchmark
    compute
)



enable_testing()

# Tests
add_executable(
  tests
  src/tests.cpp
)
target_link_libraries(tests
  gtest_main
  compute
)

gtest_discover_tests(tests PROPERTIES DISCOVERY_TIMEOUT 600)
