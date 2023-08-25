#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>

#include <solver/linear_solver.hpp>
#include <type_traits>

namespace compute {

template <typename DataType>
class LLTSolverCUDA : public LinearSolver<DataType> {
 private:
  bool first_iter;
  cusolverSpHandle_t handle;
  Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
  cusparseMatDescr_t desc;

  // matrix buffers
  DataType* data;
  int* outerIndices;
  int* innerIndices;
  DataType* b_dev;
  DataType* x_dev;
  int reorder;
  DataType tol;

 public:
  LLTSolverCUDA(int ordering_mode = 0, DataType tol = 1e-7)
      : first_iter(true),
        data(nullptr),
        outerIndices(nullptr),
        innerIndices(nullptr),
        b_dev(nullptr),
        x_dev(nullptr),
        reorder(ordering_mode),
        tol(tol) {
    static_assert(std::is_same<double, DataType>(),
                  "Only doubles supported by solver!");
    cusolverSpCreate(&handle);

    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(
        desc, CUSPARSE_FILL_MODE_LOWER);  // we generate UT CCS matrices, which
                                          // should be CSR lower when transposed
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
  }

  ~LLTSolverCUDA() {
    if (data) {
      cudaFree(data);
      cudaFree(outerIndices);
      cudaFree(innerIndices);
      cudaFree(b_dev);
      cudaFree(x_dev);
    }
    cusparseDestroyMatDescr(desc);
    cusolverSpDestroy(handle);
  }

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->fill_csc2(matrix, true);

      int nnz = matrix.nonZeros();

      cudaMalloc((void**)&data, sizeof(DataType) * nnz);
      cudaMalloc((void**)&outerIndices, sizeof(int) * (matrix.outerSize() + 1));
      cudaMalloc((void**)&innerIndices, sizeof(int) * nnz);

      cudaMalloc((void**)&b_dev, b->mem_size());
      cudaMalloc((void**)&x_dev, x->mem_size());

      cudaMemcpy(outerIndices, matrix.outerIndexPtr(),
                 sizeof(int) * (matrix.outerSize() + 1),
                 cudaMemcpyKind::cudaMemcpyDefault);
      cudaMemcpy(innerIndices, matrix.innerIndexPtr(), sizeof(int) * nnz,
                 cudaMemcpyKind::cudaMemcpyDefault);
      first_iter = false;
    } else {
      A->fill_csc_values2(matrix.valuePtr(), true);
    }
    cudaMemcpy(data, matrix.valuePtr(), sizeof(DataType) * matrix.nonZeros(),
                 cudaMemcpyKind::cudaMemcpyDefault);
    cudaMemcpy(b_dev, b->map(), b->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);

    cudaMemset(x_dev, 0, x->mem_size());

    // solve
    auto t0 = std::chrono::high_resolution_clock::now();
    int singularity = 0;

    if (!data || !b_dev || !x_dev || !outerIndices || !innerIndices) {
      std::cerr << "memory allocation failed!" << std::endl;
      return false;
    }

    auto status = cusolverSpDcsrlsvchol(
        handle, matrix.cols(), matrix.nonZeros(), desc, data, outerIndices,
        innerIndices, b_dev, tol, reorder, x_dev, &singularity);

    cudaMemcpy(x->map(), x_dev, x->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);

    auto t1 = std::chrono::high_resolution_clock::now();
    // fmt::print("cuSolver Took: {}\n", std::chrono::duration<double>(t1 -
    // t0).count());

    if (status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Error: LLTSolverCUDA returned status " << status << "!\n";
      return false;
    }

    return true;
  }

  bool result_gpu() override { return false; }
};

template <typename DataType>
class LLTSolverCUDALowLevel : public LinearSolver<DataType> {
 private:
  bool first_iter;
  cusolverSpHandle_t handle;
  Eigen::SparseMatrix<DataType, Eigen::ColMajor> matrix;
  cusparseMatDescr_t desc;
  csrcholInfo_t chol_info;

  // matrix buffers
  DataType* data;
  int* outerIndices;
  int* innerIndices;
  DataType* b_dev;
  DataType* x_dev;
  int reorder;
  size_t internal_bytes;
  size_t workspace_bytes;
  void* workspace_buffer;

 public:
  LLTSolverCUDALowLevel()
      : first_iter(true),
        data(nullptr),
        outerIndices(nullptr),
        innerIndices(nullptr),
        b_dev(nullptr),
        x_dev(nullptr),
        workspace_buffer(nullptr) {
    static_assert(std::is_same<double, DataType>(),
                  "Only doubles supported by solver!");
    cusolverSpCreate(&handle);
    cusolverSpCreateCsrcholInfo(&chol_info);

    cusparseCreateMatDescr(&desc);
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(
        desc, CUSPARSE_FILL_MODE_LOWER);  // we generate UT CCS matrices, which
                                          // should be CSR lower when transposed
    cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
  }

  ~LLTSolverCUDALowLevel() {
    if (data) {
      cudaFree(data);
      cudaFree(outerIndices);
      cudaFree(innerIndices);
      cudaFree(b_dev);
      cudaFree(x_dev);
      cudaFree(workspace_buffer);
    }
    cusparseDestroyMatDescr(desc);
    cusolverSpDestroyCsrcholInfo(chol_info);
    cusolverSpDestroy(handle);
  }

  bool solve(MatPtr<DataType> A, BufferPtr<DataType> x,
             BufferPtr<DataType> b) override {
    if (first_iter) {
      A->fill_csc2(matrix, true);

      int nnz = matrix.nonZeros();

      cudaMalloc((void**)&data, sizeof(DataType) * nnz);
      cudaMalloc((void**)&outerIndices, sizeof(int) * (matrix.outerSize() + 1));
      cudaMalloc((void**)&innerIndices, sizeof(int) * nnz);

      cudaMalloc((void**)&b_dev, b->mem_size());
      cudaMalloc((void**)&x_dev, x->mem_size());

      cudaMemcpy(outerIndices, matrix.outerIndexPtr(),
                 sizeof(int) * (matrix.outerSize() + 1),
                 cudaMemcpyKind::cudaMemcpyDefault);
      cudaMemcpy(innerIndices, matrix.innerIndexPtr(), sizeof(int) * nnz,
                 cudaMemcpyKind::cudaMemcpyDefault);

      // Analyze
      cusolverSpXcsrcholAnalysis(handle, matrix.rows(), matrix.nonZeros(), desc,
                                 outerIndices, innerIndices, chol_info);

      // Create workspace buffer
      cusolverSpDcsrcholBufferInfo(
          handle, matrix.rows(), matrix.nonZeros(), desc, data, outerIndices,
          innerIndices, chol_info, &internal_bytes, &workspace_bytes);

      cudaMalloc(&workspace_buffer, workspace_bytes);

      first_iter = false;
    } else {
      A->fill_csc_values2(matrix.valuePtr(), true);
    }
    cudaMemcpy(data, matrix.valuePtr(), sizeof(DataType) * matrix.nonZeros(),
                 cudaMemcpyKind::cudaMemcpyDefault);
    cudaMemcpy(b_dev, b->map(), b->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);
    cudaMemset(x_dev, 0, x->mem_size());

    auto t0 = std::chrono::high_resolution_clock::now();

    // solve
    cusolverSpDcsrcholFactor(handle, matrix.rows(), matrix.nonZeros(), desc,
                             data, outerIndices, innerIndices, chol_info,
                             workspace_buffer);
    auto status = cusolverSpDcsrcholSolve(handle, matrix.rows(), b_dev, x_dev,
                                          chol_info, workspace_buffer);
    cudaMemcpy(x->map(), x_dev, x->mem_size(),
               cudaMemcpyKind::cudaMemcpyDefault);

    auto t1 = std::chrono::high_resolution_clock::now();
    // fmt::print("cuSolver Took: {}\n", std::chrono::duration<double>(t1 -
    // t0).count());

    if (status != cusolverStatus_t::CUSOLVER_STATUS_SUCCESS) {
      std::cerr << "Error: LLTSolverCUDALowLevel returned status " << status
                << "!\n";
      return false;
    }

    return true;
  }

  bool result_gpu() override { return false; }
};

}  // namespace compute