#pragma once

#include "solver/gpu_buffer.hpp"
#include <hipSYCL/sycl/libkernel/group_functions.hpp>
#include <hipSYCL/sycl/libkernel/nd_item.hpp>
#include <hipSYCL/sycl/libkernel/nd_range.hpp>
#include <hipSYCL/sycl/libkernel/vec.hpp>
#include <numeric>
#include <algorithm>
#include <vector>
#include <rh/robin_hood.h>
#include <sycl/sycl.hpp>

namespace compute
{
    template <typename T>
    T ceil_div(const T &n, const T &d)
    {
        return (n + d - 1) / d;
    }

    struct ArrayInfo
    {
        uint32_t n;
    };
    // Provides functionality for recording and executing solver operations on the GPU
    class SYCLSolverSeq
    {
    private:
        SYCLComputeEngine *engine;
        bool init;
        bool init_ops; // indicates if init sequence recorded operations

        std::vector<std::function<void()>> init_sequence;
        std::vector<std::function<void()>> sequence;
        std::vector<sycl::event> events;
        sycl::queue& queue;
        // bool barrierRequested;
        std::vector<sycl::event> barrier_events;

    public:
        SYCLSolverSeq(SYCLComputeEngine *engine, sycl::queue& queue) : engine(engine), init(true), queue(queue), init_ops(false) //, barrierRequested(false)
        {
        }

        template <typename DataType>
        void inner_product(SCBPtr<DataType> v1, SCBPtr<DataType> v2, SCBPtr<DataType> out1, SCBPtr<DataType> out2)
        {
            mul_vec(v1, v2, out1);
            insert_cc_barrier();
            reduction(out1, out2);
        }

        template <typename DataType>
        void self_inner_product(SCBPtr<DataType> in, SCBPtr<DataType> out1, SCBPtr<DataType> out2)
        {
            mul_vec(in, in, out1);
            insert_cc_barrier();
            reduction(out1, out2);
        }

        // Input buffer will be modified, final result is stored in output buffer
        template <typename DataType>
        void reduction(SCBPtr<DataType> in, SCBPtr<DataType> out)
        {
            
            // // Simpler code but maybe slower?
            // sequence.push_back([=](){
            //     queue.submit([&](sycl::handler& cgh) {
            //     DataType* x = in->get_op_ptr();
            //     DataType* y = out->get_op_ptr();

            //     auto sum_reduction = sycl::reduction(y, sycl::plus<DataType>());
            //     // this is too slow
            //     cgh.parallel_for<class reduce_vec>(sycl::range<1>{in->size()}, sum_reduction,
            //                 [=](auto idx, auto& sum){
            //                     sum += x[idx];
            //             });
            //     });
            // });
            // insert_cc_barrier();
            // Alternate implementation: also seems to work, may be slightly faster
            reduce_impl_outer<DataType>(in, out);
        }

        template <typename DataType>
        void copy_vec(SCBPtr<DataType> src, SCBPtr<DataType> dest, uint32_t size = std::numeric_limits<uint32_t>::max())
        {

            sequence.push_back([=](){
                const uint32_t bsize = std::min(static_cast<uint32_t>(src->size()), size);
                // auto& queue = engine->get_queue();
                auto ev = queue.copy(src->get_op_ptr(), dest->get_op_ptr(), bsize);
                events.push_back(ev);
            });
        }

        template <typename DataType>
        void fill_vec(SCBPtr<DataType> dest, const DataType value)
        {
            sequence.push_back([=](){
                // auto & queue= engine->get_queue();
                auto ev = queue.fill(dest->get_op_ptr(), value, dest->size());
                events.push_back(ev);
            });

        }

        // w = u+a*v
        template <typename DataType>
        void add_vec(SCBPtr<DataType> u, SCBPtr<DataType> v, SCBPtr<DataType> w, DataType a)
        {

            sequence.push_back([=](){
                auto ev = queue.submit([&](sycl::handler& cgh) {
                    DataType* c = w->get_op_ptr();
                    DataType* x = u->get_op_ptr();
                    DataType* y = v->get_op_ptr();
                    DataType factor = a;

                    cgh.parallel_for<class add_vec>(sycl::range<1>{u->size()}, 
                        [=](auto idx){
                            c[idx] = x[idx] + factor*y[idx];
                        });
                });
                events.push_back(ev);
            });
        }

        // w = u+a*v
        template <typename DataType>
        void add_vec(SCBPtr<DataType> u, SCBPtr<DataType> v, SCBPtr<DataType> w, SCBPtr<DataType> a, bool add = true)
        {
            sequence.push_back([=](){
                auto ev = queue.submit([&](sycl::handler& cgh) {
                    DataType* c = w->get_op_ptr();
                    DataType* x = u->get_op_ptr();
                    DataType* y = v->get_op_ptr();
                    DataType* a_ptr = a->get_op_ptr();

                    if (add) {
                        cgh.parallel_for<class add_vec_vec>(sycl::range<1>{u->size()}, 
                                [=](auto idx){
                                    c[idx] = x[idx] + a_ptr[0]*y[idx];
                            });
                    }
                    else {
                            cgh.parallel_for<class sub_vec_vec>(sycl::range<1>{u->size()}, 
                                [=](auto idx){
                                    c[idx] = x[idx] -  a_ptr[0]*y[idx];
                            });   
                    }
 
                });
                events.push_back(ev);
            });
        }

        template <typename DataType>
        void div_vec(SCBPtr<DataType> u, SCBPtr<DataType> v,
                     SCBPtr<DataType> w, uint32_t size = std::numeric_limits<uint32_t>::max())
        {
            sequence.push_back([=](){
                auto ev = queue.submit([&](sycl::handler& cgh) {
                    DataType* c = w->get_op_ptr();
                    DataType* x = u->get_op_ptr();
                    DataType* y = v->get_op_ptr();
                    const auto buf_size = std::min(static_cast<uint32_t>(u->size()), size);

                    cgh.parallel_for<class div_vec>(sycl::range<1>{buf_size}, 
                        [=](auto idx){
                            c[idx] = x[idx] / y[idx];
                        });
                });
                events.push_back(ev);
            });
        }



        // Inserts a compute-compute barrier
        void insert_cc_barrier()
        {
            insert_wait();
        }

        // Inserts a transfer-compute barrier
        void insert_tc_barrier()
        {
            insert_wait();
        }

        // Inserts a compute-transfer barrier
        void insert_ct_barrier()
        {
            insert_wait();
        }

        struct MultiplyInfo
        {
            uint32_t start;
            uint32_t n;
        };

        // TODO: Move!
        uint32_t distribute(uint32_t & start, uint32_t work, uint32_t num_groups, uint32_t id) {
            uint32_t items = work / num_groups;
            work -= items*num_groups;
            start += items*id;

            // evenly distribute the remainder as much as possible
            if (work > 0) {
                if (id < work) {
                    items++;
                }
                start += sycl::min(id, work); // if something was added, shift up by number of groups before
            }
            return items;
        }

        template <typename DataType>
        void multiply_large(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                            SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest,
                            SCBPtr<uint32_t> lists, SCBPtr<uint32_t> pairs,
                            bool add = true, bool transpose_right = false)
        {



            


            sequence.push_back([=](){


            struct MultiplyInfo
            {
                uint32_t start;
                uint32_t n;
            };

                struct SpecConstants {
                uint32_t rows_a;
                uint32_t cols_a;
                uint32_t rows_b;
                uint32_t cols_b;
                uint32_t add;
                uint32_t transpose_right;
                uint32_t num_mat;
                // uint32_t szc;
            };

            // assume max workgroup size of 1024
            constexpr uint32_t max_size = 512;
            const uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            const auto subgroup_size = engine->get_subgroup_size();


            if (szc > max_size)
            {
                throw std::runtime_error("Output matrix size is greater than max workgroup size!");
            }
            // num subgroups per output matrix
            int32_t sg_per_mat = static_cast<int32_t>(ceil_div(szc, subgroup_size));

            int32_t mat_per_wg = static_cast<int32_t>(max_size / (subgroup_size * sg_per_mat)); // workload division factor

            int32_t num_subgroups = mat_per_wg * sg_per_mat;
            int32_t local_size_x = num_subgroups * subgroup_size;


                // auto & queue= engine->get_queue();

                const MulList * p_lists = reinterpret_cast<MulList*>(lists->get_op_ptr());
                const sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
                const DataType* a = left->get_op_ptr();
                const DataType* b = right->get_op_ptr();
                DataType* c = dest->get_op_ptr();




                const auto rows_a = dim1.first;
                const auto cols_a = dim1.second;
                const auto rows_b = dim2.first;
                const auto cols_b = dim2.second;
                const auto num_mat = static_cast<uint32_t>(mat_per_wg);

                // const SpecConstants sc{rows_a, cols_a, rows_b, cols_b, static_cast<uint32_t>(add), static_cast<uint32_t>(transpose_right), num_mat};

                // constexpr sycl::specialization_id<SpecConstants> coeff_id;
                // std::cout << "n: " << n << std::endl;
                const MultiplyInfo ml_info{start, n};
                auto ev = queue.submit([&](sycl::handler& cgh) {
                                    


                sycl::accessor <DataType, 2, sycl::access::mode::read_write, sycl::access::target::local>
                                        values(sycl::range<2>(num_mat, szc), cgh);
                    // auto wait_list = queue.get_wait_list();
                    // cgh.depends_on(wait_list);
                    
                    cgh.parallel_for<class multiply_large>(sycl::nd_range{sycl::range<1>{n*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){


     

                        auto sg = it.get_sub_group();
                        const auto sub_group_invocation_id = sg.get_local_linear_id();
                        const auto sub_group_id = sg.get_group_linear_id();

                        const auto ml_idx = it.get_group_linear_id();
                        const auto elem = sub_group_invocation_id + (sub_group_id % sg_per_mat)*subgroup_size;
                        const auto mat_id = sub_group_id/sg_per_mat;
                        uint32_t off_c;
                        if (ml_idx < ml_info.n && elem < szc) {
                            const auto list_id = ml_info.start + ml_idx;
                            uint32_t start = p_lists[list_id].start;
                            uint32_t items = distribute(start, p_lists[list_id].n, num_subgroups/sg_per_mat, mat_id);
                            off_c = p_lists[list_id].dest;

                            const uint32_t c_row = elem % rows_a;
                            const uint32_t c_col = elem / rows_a;

                            double value = 0.0;
                            const uint32_t end = start + items;
                            for (uint32_t i = start; i < end; i++) {
                                const sycl::uint2 pair = p_pairs[i];
                                // value += mulAB(pair.x(), pair.y(), c_row, c_col);
                                auto off_a = pair.x();
                                auto off_b = pair.y();

                                if (transpose_right) {
                                    for (int v = 0; v < cols_a; v++) {
                                        value += a[c_row+v*rows_a + off_a]*b[v*cols_b+c_col + off_b];
                                    }
                                }
                                else {
                                    for (int v = 0; v < cols_a; v++) {
                                        value += a[c_row+v*rows_a + off_a]*b[v+c_col*rows_b + off_b];
                                    }
                                }


                            }
                            values[mat_id][elem] = value;
                        }
                        // group_barrier(it.get_group(), sycl::memory_scope::work_group);
                        group_barrier(it.get_group());
                        // it.barrier();

                        if (ml_idx < ml_info.n && elem < szc && mat_id == 0) {

                            double sum = 0.0;
                            for (uint i = 0; i < num_mat; i++) {
                                sum += values[i][elem];
                            }

                            if (!add) {
                                sum = -sum; 
                            }
                            c[elem + off_c] += sum;  
                        }    

                    });

                });
                events.push_back(ev);
            });
        }

        template <typename DataType>
        void multiply(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                      SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest,
                      SCBPtr<uint32_t> lists, SCBPtr<uint32_t> pairs,
                      bool add = true, bool transpose_right = false)
        {
            multiply_large(dim1, dim2, start, n, left, right, dest, lists, pairs, add, transpose_right);
        }

        // template <typename DataType>
        // void multiply_small(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
        //                     SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest,
        //                     SCBPtr<uint32_t> lists, SCBPtr<uint32_t> pairs,
        //                     bool add = true, bool transpose_right = false)
        // {
        //     uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
        //     const auto subgroup_size = engine->get_subgroup_size();

        //     // subgroups per C
        //     auto subgroups_per_c = szc / subgroup_size;
        //     if (subgroup_size * subgroups_per_c < szc)
        //     {
        //         subgroups_per_c++;
        //     }
        //     int32_t local_size_x = subgroups_per_c * subgroup_size;
        //     // auto tc0 = std::chrono::high_resolution_clock::now();
        //     std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0});

        //     const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

        //     kp::Manager &mgr = *(engine->get_manager());

        //     auto algo = mgr.algorithm<int32_t, MultiplyInfo>(
        //         tensors,
        //         engine->spv_sbm_multiply_subgroup, {n, 1, 1}, spec, {MultiplyInfo{start, n}});

        //     seq->record<kp::OpAlgoDispatch>(algo);
        // }

        template <typename DataType>
        void multiply_packed(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                             SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest,
                             SCBPtr<uint32_t> lists, SCBPtr<uint32_t> pairs,
                             bool add = true, bool transpose_right = false)
        {

            sequence.push_back([=, this](){

                const int32_t local_size_x = 256;
                const auto szc = static_cast<uint32_t>(dim1.first * dim2.second);
                const uint32_t wgx = ceil_div(n * szc, static_cast<uint32_t>(local_size_x));

                // auto & queue= engine->get_queue();


                MulList * p_lists = reinterpret_cast<MulList*>(lists->get_op_ptr());
                sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
                DataType* a = left->get_op_ptr();
                DataType* b = right->get_op_ptr();
                DataType* c = dest->get_op_ptr();




                const auto rows_a = dim1.first;
                const auto cols_a = dim1.second;
                const auto rows_b = dim2.first;
                const auto cols_b = dim2.second;
                const auto num_lists = n;
                const auto start_idx = start;


                const MultiplyInfo ml_info{start, n};
                auto ev = queue.submit([&](sycl::handler& cgh) {
                                    
                    cgh.parallel_for<class multiply_packed>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){

                        const auto ml_idx = it.get_global_linear_id() / szc;
                        const auto elem = it.get_global_linear_id() % szc;

                        if (!(ml_idx < num_lists && elem < szc)) {
                            return;
                        }

                        const uint32_t list_idx = start_idx + ml_idx;
                        const uint32_t start = p_lists[list_idx].start;
                        const uint32_t items = p_lists[list_idx].n;
                        const uint32_t off_c = p_lists[list_idx].dest;

                        const uint32_t c_row = elem % rows_a;
                        const uint32_t c_col = elem / rows_a;

                        double value = 0.0;
                        const uint32_t end = start + items;
                        for (uint32_t i = start; i < end; i++) {
                            const sycl::uint2 pair = p_pairs[i];
                            // value += mulAB(pair.x(), pair.y(), c_row, c_col);
                            auto off_a = pair.x();
                            auto off_b = pair.y();

                            if (transpose_right) {
                                for (int v = 0; v < cols_a; v++) {
                                    value += a[c_row+v*rows_a + off_a]*b[v*cols_b+c_col + off_b];
                                }
                            }
                            else {
                                for (int v = 0; v < cols_a; v++) {
                                    value += a[c_row+v*rows_a + off_a]*b[v+c_col*rows_b + off_b];
                                }
                            }
                        }

                        if (!add) {
                            value = -value;
                        }
                        c[c_row + c_col*rows_a + off_c] += value;

                    });

                });
                events.push_back(ev);

            });


        }

        template <typename DataType>
        void multiply_abat(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                           SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest,
                           SCBPtr<uint32_t> lists, SCBPtr<uint32_t> pairs,
                           bool add = true)
        {



            // std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0});

            // const std::vector<std::shared_ptr<kp::Tensor>> tensors{left->get_tensor(), right->get_tensor(), dest->get_tensor(), lists->get_tensor(), pairs->get_tensor()};

            // kp::Manager &mgr = *(engine->get_manager());

            // auto algo = mgr.algorithm<int32_t, MultiplyInfo>(
            //     tensors,
            //     engine->spv_sbm_multiply_abat, {n, 1, 1}, spec, {MultiplyInfo{start, n}});

            // seq->record<kp::OpAlgoDispatch>(algo);

            sequence.push_back([=](){

                // uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
                const auto subgroup_size = engine->get_subgroup_size();
                // const auto num_subgroups = local_size_x / subgroup_size;

                // subgroups per C
                int32_t req_threads = std::max(dim1.first * dim2.second, dim1.first * dim1.first);
                // int32_t subgroups_per_c = (req_threads - 1 + subgroup_size) / subgroup_size;
                int32_t subgroups_per_c = ceil_div<int32_t>(req_threads, subgroup_size);


                const int32_t local_size_x = subgroups_per_c * subgroup_size;

                const auto szc = static_cast<uint32_t>(dim1.first * dim1.first);
                const auto szab = static_cast<uint32_t>(dim1.first * dim2.second);
                // const uint32_t wgx = ceil_div(n * szc, static_cast<uint32_t>(local_size_x));


                // auto & queue= engine->get_queue();

                MulList * p_lists = reinterpret_cast<MulList*>(lists->get_op_ptr());
                sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
                DataType* a = left->get_op_ptr();
                DataType* b = right->get_op_ptr();
                DataType* c = dest->get_op_ptr();




                const auto rows_a = dim1.first;
                const auto cols_a = dim1.second;
                const auto rows_b = dim2.first;
                const auto cols_b = dim2.second;
                // const auto num_lists = n;
                // const auto start_idx = start;
                

                const MultiplyInfo ml_info{start, n};
                auto ev = queue.submit([&](sycl::handler& cgh) {
                                    

                    sycl::accessor <DataType, 2, sycl::access::mode::read_write, sycl::access::target::local>
                                        mat_ab(sycl::range<2>(rows_a, cols_b), cgh);


                    cgh.parallel_for<class multiply_abat>(sycl::nd_range{sycl::range<1>{n*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){

                        const auto ml_idx = it.get_group_linear_id();
                        const auto elem = it.get_local_linear_id();

                        const uint32_t list_idx = ml_info.start + ml_idx;
                        const uint32_t start = p_lists[list_idx].start;
                        const uint32_t items = p_lists[list_idx].n;
                        const uint32_t off_c = p_lists[list_idx].dest;

                        const uint c_row = elem % rows_a;
                        const uint c_col = elem / rows_a;

                        const uint a_row = c_row;
                        const uint b_col = c_col;

                        DataType value = 0;
                        for (uint32_t i = 0; i < items; i++) {
                            const sycl::uint2 pair = p_pairs[start+i];
                            const auto off_a = pair.x();
                            const auto off_b = pair.y();

                            if (elem < szab) {
                                // mulAB
                                DataType ab = 0;
                                for (int v = 0; v < cols_a; v++) {
                                    ab += a[a_row+v*rows_a + off_a]*b[v+b_col*rows_b + off_b];
                                }
                                mat_ab[a_row][b_col] = ab;
                            }
                    
                            // group_barrier(it.get_group(), sycl::memory_scope::work_group);
                            group_barrier(it.get_group());
                            // it.barrier();
                            if (elem < szc) {
                                // mulABAT
                                for (int w = 0; w < cols_b; w++) {
                                    value += mat_ab[c_row][w]*a[c_col+w*rows_a + off_a];
                                }       
                            }
                            // group_barrier(it.get_group(), sycl::memory_scope::work_group);
                            // it.barrier();
                            group_barrier(it.get_group());
                        }


                        if (elem < szc) {
                            if (!add) {
                                value = -value;
                            }

                            c[c_row + c_col*rows_a + off_c] += value;
                        }
                    });

                });
                events.push_back(ev);

            });
        }

        template <typename DataType>
        void tmultiply(
            BlockDim dim1, BlockDim dim2, uint32_t num_blocks,
            SCBPtr<uint32_t> dest_idx,
            SCBPtr<DataType> dest_data,
            SCBPtr<uint32_t> left_idx,
            SCBPtr<uint32_t> left_offsets,
            SCBPtr<uint32_t> left_ptr,
            SCBPtr<DataType> left_data,
            SCBPtr<uint32_t> right_idx,
            SCBPtr<uint32_t> right_offsets,
            SCBPtr<uint32_t> right_ptr,
            SCBPtr<DataType> right_data,
            bool add = true,
            bool transpose_right = false)
        {

            // // get or create algorithm for these specific tensors

            // uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            // const auto subgroup_size = engine->get_subgroup_size();

            // struct MultiplyInfo
            // {
            //     uint32_t start;
            //     uint32_t n;
            // };

            // // assume some max workgroup / matrix size
            // constexpr uint32_t max_size = 256;

            // if (szc > max_size)
            // {
            //     throw std::runtime_error("Output matrix size is greater than max workgroup size!");
            // }
            // // num subgroups per output matrix
            // int32_t sg_per_mat = static_cast<int32_t>(ceil_div(szc, subgroup_size));
            // int32_t mat_per_wg = static_cast<int32_t>(max_size / (subgroup_size * sg_per_mat)); // workload division factor

            // int32_t num_subgroups = mat_per_wg * sg_per_mat;
            // int32_t local_size_x = num_subgroups * subgroup_size;
            // std::vector<int32_t> spec({dim1.first, dim1.second, dim2.first, dim2.second, local_size_x, add ? 1 : 0, transpose_right ? 1 : 0, sg_per_mat, mat_per_wg});
            // const std::vector<std::shared_ptr<kp::Tensor>> tensors{
            //     dest_idx->get_tensor(),
            //     dest_data->get_tensor(),
            //     left_idx->get_tensor(),
            //     left_offsets->get_tensor(),
            //     left_ptr->get_tensor(),
            //     left_data->get_tensor(),
            //     right_idx->get_tensor(),
            //     right_offsets->get_tensor(),
            //     right_ptr->get_tensor(),
            //     right_data->get_tensor(),
            // };

            // kp::Manager &mgr = *(engine->get_manager());
            // auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
            //                                                  engine->spv_sbm_multiply_dynamic, {}, spec, {MultiplyInfo{0, num_blocks}});

            // // algo->setWorkgroup(kp::Workgroup{n, 1, 1});
            // algo->setWorkgroup(kp::Workgroup{num_blocks, 1, 1});

            // // Dispatch until all output matrix elements have been handled
            // // MultiplyInfo push_constant{start, n};
            // // algo->setPushConstants(&push_constant, 1, sizeof(push_constant));
            // seq->record<kp::OpAlgoDispatch>(algo);
            // // std::cout << "Algo dispatched!\n";
            // // std::cout << "num_blocks: " << num_blocks << std::endl;

            // sequence.push_back([=, this](){


            //     uint32_t szc = static_cast<uint32_t>(dim1.first * dim2.second);
            //     const auto subgroup_size = engine->get_subgroup_size();

            //     struct MultiplyInfo
            //     {
            //         uint32_t start;
            //         uint32_t n;
            //     };

            //     // assume some max workgroup / matrix size
            //     constexpr uint32_t max_size = 256;

            //     if (szc > max_size)
            //     {
            //         throw std::runtime_error("Output matrix size is greater than max workgroup size!");
            //     }
                
            //     // num subgroups per output matrix
            //     int32_t sg_per_mat = static_cast<int32_t>(ceil_div(szc, subgroup_size));
            //     int32_t mat_per_wg = static_cast<int32_t>(max_size / (subgroup_size * sg_per_mat)); // workload division factor

            //     int32_t num_subgroups = mat_per_wg * sg_per_mat;
            //     int32_t local_size_x = num_subgroups * subgroup_size;

            //     auto & queue= engine->get_queue();


            //     MulList * p_lists = reinterpret_cast<MulList*>(lists->get_op_ptr());
            //     sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
            //     DataType* a = left->get_op_ptr();
            //     DataType* b = right->get_op_ptr();
            //     DataType* c = dest->get_op_ptr();




            //     const auto rows_a = dim1.first;
            //     const auto cols_a = dim1.second;
            //     const auto rows_b = dim2.first;
            //     const auto cols_b = dim2.second;
            //     const auto num_lists = n;
            //     const auto start_idx = start;


            //     const MultiplyInfo ml_info{start, n};
            //     queue.submit([&](sycl::handler& cgh) {
                                    
            //         cgh.parallel_for<class multiply_packed>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
            //         [=](sycl::nd_item<1> it){

            //             const auto ml_idx = it.get_global_linear_id() / szc;
            //             const auto elem = it.get_global_linear_id() % szc;

            //             if (!(ml_idx < num_lists && elem < szc)) {
            //                 return;
            //             }

            //             const uint32_t list_idx = start_idx + ml_idx;
            //             const uint32_t start = p_lists[list_idx].start;
            //             const uint32_t items = p_lists[list_idx].n;
            //             const uint32_t off_c = p_lists[list_idx].dest;

            //             const uint32_t c_row = elem % rows_a;
            //             const uint32_t c_col = elem / rows_a;

            //             double value = 0.0;
            //             const uint32_t end = start + items;
            //             for (uint32_t i = start; i < end; i++) {
            //                 const sycl::uint2 pair = p_pairs[i];
            //                 // value += mulAB(pair.x(), pair.y(), c_row, c_col);
            //                 auto off_a = pair.x();
            //                 auto off_b = pair.y();

            //                 if (transpose_right) {
            //                     for (int v = 0; v < cols_a; v++) {
            //                         value += a[c_row+v*rows_a + off_a]*b[v*cols_b+c_col + off_b];
            //                     }
            //                 }
            //                 else {
            //                     for (int v = 0; v < cols_a; v++) {
            //                         value += a[c_row+v*rows_a + off_a]*b[v+c_col*rows_b + off_b];
            //                     }
            //                 }
            //             }

            //             if (!add) {
            //                 value = -value;
            //             }
            //             c[c_row + c_col*rows_a + off_c] += value;

            //         });

            //     });

            // });
        }

        template <typename DataType>
        void estimate_list(
            uint32_t num_blocks,
            SCBPtr<uint32_t> dest_idx,
            SCBPtr<uint32_t> left_idx,
            SCBPtr<uint32_t> left_offsets,
            SCBPtr<uint32_t> left_ptr,
            SCBPtr<uint32_t> right_idx,
            SCBPtr<uint32_t> right_offsets,
            SCBPtr<uint32_t> right_ptr,
            SCBPtr<uint32_t> allocator_info)
        {

            // // get or create algorithm for these specific tensors
            // struct MultiplyInfo
            // {
            //     uint32_t n;
            // };

            // const int32_t local_size = static_cast<int32_t>(engine->get_subgroup_size());
            // std::vector<int32_t> spec({local_size});

            // const std::vector<std::shared_ptr<kp::Tensor>> tensors{
            //     dest_idx->get_tensor(),
            //     left_idx->get_tensor(), left_offsets->get_tensor(), left_ptr->get_tensor(),
            //     right_idx->get_tensor(), right_offsets->get_tensor(), right_ptr->get_tensor(),
            //     allocator_info->get_tensor()};

            // kp::Manager &mgr = engine->get_manager_ref();
            // auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
            //                                                  engine->spv_sbm_multiply_estimate_list, {}, spec, {MultiplyInfo{num_blocks}});

            // const uint32_t wgx = (num_blocks + local_size - 1) / local_size;
            // algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});

            // // Dispatch until all output matrix elements have been handled
            // seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void build_list(
            uint32_t num_blocks,
            SCBPtr<uint32_t> dest_idx,
            SCBPtr<uint32_t> left_idx,
            SCBPtr<uint32_t> left_offsets,
            SCBPtr<uint32_t> left_ptr,
            SCBPtr<uint32_t> right_idx,
            SCBPtr<uint32_t> right_offsets,
            SCBPtr<uint32_t> right_ptr,
            SCBPtr<uint32_t> mul_lists,
            SCBPtr<uint32_t> mul_pairs,
            SCBPtr<uint32_t> allocator_info)
        {

            // // get or create algorithm for these specific tensors

            // struct MultiplyInfo
            // {
            //     uint32_t num_blocks;
            // };

            // const int32_t local_size = static_cast<int32_t>(engine->get_subgroup_size());

            // std::vector<int32_t> spec({local_size});

            // const std::vector<std::shared_ptr<kp::Tensor>> tensors{
            //     dest_idx->get_tensor(),
            //     left_idx->get_tensor(), left_offsets->get_tensor(), left_ptr->get_tensor(),
            //     right_idx->get_tensor(), right_offsets->get_tensor(), right_ptr->get_tensor(),
            //     mul_lists->get_tensor(), mul_pairs->get_tensor(),
            //     allocator_info->get_tensor()};

            // kp::Manager &mgr = *(engine->get_manager());
            // auto algo = mgr.algorithm<int32_t, MultiplyInfo>(tensors,
            //                                                  engine->spv_sbm_multiply_build_list, {}, spec, {MultiplyInfo{num_blocks}});

            // const uint32_t wgx = (num_blocks + local_size - 1) / local_size;
            // algo->setWorkgroup(kp::Workgroup{wgx, 1, 1});

            // // Dispatch until all output matrix elements have been handled
            // seq->record<kp::OpAlgoDispatch>(algo);
        }

        template <typename DataType>
        void dispatch_right_block_diagonal(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                                           SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest, SCBPtr<uint32_t> pairs, bool add)
        {

            sequence.push_back([=, this](){

                const uint32_t local_size_x = static_cast<int32_t>(engine->get_subgroup_size());
                const uint32_t szc = dim1.first * dim2.second;
                const uint32_t wgx = (szc * n + local_size_x - 1) / local_size_x;

                // auto & queue= engine->get_queue();


                sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
                DataType* a = left->get_op_ptr();
                DataType* b = right->get_op_ptr();
                DataType* c = dest->get_op_ptr();




                const auto rows_a = dim1.first;
                const auto cols_a = dim1.second;
                const auto rows_b = dim2.first;
                const auto cols_b = dim2.second;
                const auto num_lists = n;
                const auto start_idx = start;

                const MultiplyInfo ml_info{start, n};
                auto ev = queue.submit([&](sycl::handler& cgh) {
                                    
                    cgh.parallel_for<class multiply_right_block_diagonal>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){

                        const auto idx = it.get_global_linear_id() / szc;
                        if (idx >= ml_info.n) {
                            return;
                        }

                        const auto elem = it.get_global_linear_id() % szc;
                        const uint32_t c_row = elem % rows_a;
                        const uint32_t c_col = elem / rows_a;


                        const sycl::uint2 pair = p_pairs[ml_info.start + idx];
                        auto off_a = pair.x();
                        auto off_b = pair.y();
                        double value = 0.0;

                        for (int v = 0; v < cols_a; v++) {
                            value += a[c_row+v*rows_a + off_a]*b[v+c_col*rows_b + off_b];
                        }

                        c[c_row + c_col*rows_a + pair.x()]  =  (add ? value : -value);

                    });

                });
                events.push_back(ev);

            });

        }

        template <typename DataType>
        void dispatch_left_block_diagonal(BlockDim dim1, BlockDim dim2, uint32_t start, uint32_t n,
                                          SCBPtr<DataType> left, SCBPtr<DataType> right, SCBPtr<DataType> dest, SCBPtr<uint32_t> pairs, bool add)
        {
                            sequence.push_back([=](){

                            const uint32_t local_size_x = static_cast<int32_t>(engine->get_subgroup_size());
                            const uint32_t szc = dim1.first * dim2.second;
                            const uint32_t wgx = (szc * n + local_size_x - 1) / local_size_x;

                            // auto & queue= engine->get_queue();


                            sycl::uint2 * p_pairs = reinterpret_cast<sycl::uint2*>(pairs->get_op_ptr());
                            DataType* a = left->get_op_ptr();
                            DataType* b = right->get_op_ptr();
                            DataType* c = dest->get_op_ptr();

                            const auto rows_a = dim1.first;
                            const auto cols_a = dim1.second;
                            const auto rows_b = dim2.first;
                            const auto cols_b = dim2.second;

                            const MultiplyInfo ml_info{start, n};
                            auto ev = queue.submit([&](sycl::handler& cgh) {
                                                
                                cgh.parallel_for<class multiply_left_block_diagonal>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                                [=](sycl::nd_item<1> it){

                                    const auto idx = it.get_global_linear_id() / szc;
                                    if (idx >= ml_info.n) {
                                        return;
                                    }

                                    const auto elem = it.get_global_linear_id() % szc;
                                    const uint32_t c_row = elem % rows_a;
                                    const uint32_t c_col = elem / rows_a;


                                    const sycl::uint2 pair = p_pairs[ml_info.start + idx];
                                    auto off_a = pair.x();
                                    auto off_b = pair.y();
                                    double value = 0.0;

                                    for (int v = 0; v < cols_a; v++) {
                                        value += a[c_row+v*rows_a + off_a]*b[v+c_col*rows_b + off_b];
                                    }

                                    // only difference is it uses pair.x() instead of pair.y()
                                    // TODO: refactor into common code
                                    c[c_row + c_col*rows_a + pair.y()]  =  (add ? value : -value);

                                });

                            });
                events.push_back(ev);

                        });
        }

        

        template <typename DataType>
        void invert_block_diagonal(SCBPtr<DataType> dest, SCBPtr<DataType> src, const robin_hood::unordered_map<Dim, std::vector<uint32_t>> &offsets)
        {

            // create a buffer to hold all offsets
            size_t buf_size = 0;
            for (auto &it : offsets)
            {
                buf_size += it.second.size();
            }

            auto buffer_type = src->get_buffer_type() == BufferType::Host ? BufferType::Host : BufferType::DeviceCached;
            auto buf_offsets = (*engine).template create_buffer<uint32_t>(nullptr, buf_size, buffer_type);
            sync_device<uint32_t>({buf_offsets}, {}, true);

            auto offset_ptr = buf_offsets->map();

            std::atomic_uint32_t write_idx(0);
            for (auto &it : offsets)
            {
                auto start_idx = write_idx.fetch_add(static_cast<uint32_t>(it.second.size()));
                auto write_ptr = reinterpret_cast<uint32_t *>(offset_ptr + start_idx);
                std::copy(it.second.begin(), it.second.end(), write_ptr);

                // create algorithm
                // auto &mgr = engine->get_manager_ref();
                // auto inv_algo = mgr.template algorithm<int32_t, MultiplyInfo>({src->get_tensor(), dest->get_tensor(), buf_offsets->get_tensor()},
                //                                                               engine->spv_sbm_bdi, {}, {it.first, local_size_x}, {MultiplyInfo{0, 0}});

                auto n = static_cast<uint32_t>(it.second.size());
                // set push constants and workgroup

                // inv_algo->setPushConstants(&push_constant, 1, sizeof(push_constant));
                // inv_algo->setWorkgroup(kp::Workgroup{num_workgroups, 1, 1});
                // seq->record<kp::OpAlgoDispatch>(inv_algo);

                    sequence.push_back([=, this](){
                        const auto local_size_x = static_cast<uint32_t>(engine->get_subgroup_size() * 8);
                        const auto num_workgroups = static_cast<uint32_t>(ceil_div(n, local_size_x));
                        const MultiplyInfo info{static_cast<uint32_t>(start_idx), n};
                        const auto rows = it.first;
      
                        // DataType* values = blocks->get_op_ptr();
                        uint32_t* offset_list = buf_offsets->get_op_ptr();
                        DataType* c = dest->get_op_ptr();
                        DataType* a = src->get_op_ptr();
     
                    

                            // auto ev = queue.submit([&](sycl::handler& cgh) {
                            auto ev = queue.submit([&](sycl::handler& cgh) {

                                                    
                                    cgh.parallel_for<class invert_block_diagonal>(sycl::nd_range{sycl::range<1>{num_workgroups*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                                    [=](sycl::nd_item<1> it){
                                        const uint32_t idx = it.get_global_linear_id();
                                        if (idx >= info.n) {
                                            return;
                                        }
                                        const uint32_t i = offset_list[info.start + idx];
                                        Eigen::Map<Eigen::Matrix2d> m2_src(a + i);
                                        Eigen::Map<Eigen::Matrix2d> m2_dest(c + i);

                                        Eigen::Map<Eigen::Matrix3d> m3_src(a + i);
                                        Eigen::Map<Eigen::Matrix3d> m3_dest(c + i);

                                        // can we use separate kernels?
                                        Eigen::Map<Eigen::Matrix4d> m4_src(a + i);
                                        Eigen::Map<Eigen::Matrix4d> m4_dest(c + i);

                                        switch (rows) {
                                            case 1:
                                                c[i] = 1.0/a[i];
                                                break;
                                            case 2:
                                                m2_dest = m2_src.inverse();
                                                break;
                                            case 3:
                                                m3_dest = m3_src.inverse();
                                                break;
                                            case 4:
                                                m4_dest = m4_src.inverse();
                                                break;
                                            default:
                                                break;
                                        }

                                    });

                                });
                events.push_back(ev);

                    });

            }

            // add final barrier
            insert_cc_barrier();
        }

        // sync device
        template <typename DataType>
        void sync_device(const std::vector<SCBPtr<DataType>> &buffers, const std::vector<std::pair<uint32_t, uint32_t>> &ranges = {}, bool once = false)
        {


            if (once) {
                init_ops = true;
            }
            // Filter these
            std::vector<std::function<void()>> & seq = once ? init_sequence : sequence;

             seq.push_back([=]() {

                            // auto & queue = engine->get_queue();
                            for (size_t i = 0; i < buffers.size(); i++)
                                {
                                    SCBPtr<DataType> b = buffers[i];
                                    if (b->get_buffer_type() == BufferType::Device || b->get_buffer_type() == BufferType::DeviceCached)
                                    {
                                        size_t start_offset = 0;
                                        size_t end_offset = b->size();
                                        if (ranges.size() > 0) {
                                            auto & r = ranges.at(i);
                                            start_offset = r.first;
                                            end_offset = r.second;
                                        }
                                        auto ev = queue.copy(b->map()+start_offset, b->get_op_ptr()+start_offset, end_offset - start_offset);
                                        events.push_back(ev);
                                    }
                                }
                });

            // Is this needed?
            seq.push_back([this](){
                // queue.wait();
                queue.wait();
                // insert_wait();
            });

        }

        template <typename DataType>
        void sync_local(const std::vector<SCBPtr<DataType>> &buffers, const std::vector<std::pair<uint32_t, uint32_t>> &ranges = {}, bool once = false)
        {
            // Filter these
            // TODO: Refactor

            if (once) {
                init_ops = true;
            }
            std::vector<std::function<void()>> & seq = once ? init_sequence : sequence;

            seq.push_back([this](){
                // queue.wait();
                queue.wait();
                // insert_wait();
            });

             seq.push_back(
                // [engine=engine, buffers, ranges, once=once]
                [=]() {
                            // auto & queue = engine->get_queue();
                            for (size_t i = 0; i < buffers.size(); i++)
                                {
                                    SCBPtr<DataType> b = buffers[i];
                                    if (b->get_buffer_type() == BufferType::Device || b->get_buffer_type() == BufferType::DeviceCached)
                                    {
                                        size_t start_offset = 0;
                                        size_t end_offset = b->size();
                                        if (ranges.size() > 0) {
                                            auto & r = ranges.at(i);
                                            start_offset = r.first;
                                            end_offset = r.second;
                                        }
                                        auto ev = queue.copy(b->get_op_ptr()+start_offset, b->map()+start_offset, end_offset - start_offset);

                                        events.push_back(ev);   

                                    }
                                }
                });

            seq.push_back([this](){
                // queue.wait();
                queue.wait();
                // insert_wait();
            });

        }

        template <typename DataType>
        void set_lambda(uint32_t dim, SCBPtr<DataType> blocks, SCBPtr<DataType> backups,
                        SCBPtr<uint32_t> offsets, SCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset)
        {
            set_restore_lambda<DataType>(dim, blocks, backups, offsets, lambda, num_mat, id_offset, backup_offset, true);
        }

        template <typename DataType>
        void restore_lambda(uint32_t dim, SCBPtr<DataType> blocks, SCBPtr<DataType> backups,
                            SCBPtr<uint32_t> offsets, SCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset)
        {
            set_restore_lambda<DataType>(dim, blocks, backups, offsets, lambda, num_mat, id_offset, backup_offset, false);
        }

        template <typename DataType>
        void copy_blocks(BlockDim dim, SCBPtr<DataType> src, SCBPtr<DataType> dest,
                         SCBPtr<uint32_t> offsets, uint32_t num_mat, uint start)
        {

            sequence.push_back([=, this](){

                const uint32_t local_size_x = 256;
                const uint32_t items = num_mat * dim.first * dim.second;
                const uint32_t wgx = ceil_div(items, local_size_x);

                DataType* p_src = src->get_op_ptr();
                DataType* p_dest = dest->get_op_ptr();
                sycl::uint2* offset_list = reinterpret_cast<sycl::uint2*>(offsets->get_op_ptr());
                // const DataType df = lambda->map()[0]; // get the damping factor - API should be redesigned so this can be avoided...
                const uint32_t start_id = start;
                const uint32_t num_elements = dim.first*dim.second;
                const uint32_t num_blocks = num_mat;
            

            // auto ev = queue.submit([&](sycl::handler& cgh) {
            auto ev = queue.submit([&](sycl::handler& cgh) {                                    
                    cgh.parallel_for<class copy_blocks_into>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){

                        const uint32_t block_id = it.get_global_linear_id()/num_elements;

                        if (block_id < num_blocks) {
                            const uint32_t element = it.get_global_linear_id() % num_elements;
                            const sycl::uint2 rw_pair = offset_list[block_id+start_id] + element;
                            p_dest[rw_pair.y()] = p_src[rw_pair.x()];
                        }


                    });

                });
                events.push_back(ev);

            });

        }

    private:

        template <typename DataType>
        void mul_vec(SCBPtr<DataType> u, SCBPtr<DataType> v,
                     SCBPtr<DataType> w, uint32_t size = std::numeric_limits<uint32_t>::max())
        {
            sequence.push_back([=](){
                auto ev  = queue.submit([&](sycl::handler& cgh) {
                    DataType* c = w->get_op_ptr();
                    DataType* x = u->get_op_ptr();
                    DataType* y = v->get_op_ptr();
                    const auto buf_size = std::min(static_cast<uint32_t>(u->size()), size);

                    cgh.parallel_for<class mul_vec>(sycl::range<1>{buf_size}, 
                        [=](auto idx){
                            c[idx] = x[idx] * y[idx];
                        });
                });
                                events.push_back(ev);

            });
        }

        template <typename DataType>
        void reduce_impl_outer(SCBPtr<DataType> in, SCBPtr<DataType> out)
        {

            constexpr int32_t local_size_x = 256;

            // Provide number of subgroups as a compile-time constant for shared mem array size

            const auto subgroups_per_workgroup = ceil_div<int32_t>(local_size_x, engine->get_subgroup_size());

            
            // record command for each level of reduction
            uint32_t n = in->size(); // number of elements at each level of reduction
            uint32_t max_iter = 4;
            bool last_dest_out = false;
            while (n > 1)
            {
                // Update number of workgroups
                const uint32_t wgx = ceil_div<uint32_t>(n, local_size_x * max_iter);

                // Record dispatch
                reduce_impl<DataType, local_size_x>(in, out, n, wgx);
                insert_cc_barrier();
                last_dest_out = !last_dest_out;
                // update n
                n = wgx;

                if (n > 1)
                {
                    // reduce back into input buffer
                    const uint32_t wgx2 = ceil_div<uint32_t>(n, local_size_x * max_iter);
                    reduce_impl<DataType, local_size_x>(out, in, n, wgx2);

                    // Pipeline barrier (before next compute dispatch)
                    insert_cc_barrier();
                    n = wgx2;
                    last_dest_out = !last_dest_out;
                }
            }

            if (!last_dest_out)
            {
                const uint32_t wgx = ceil_div<uint32_t>(n, local_size_x * max_iter);
                // Record dispatch
                reduce_impl<DataType, local_size_x>(in, out, n, wgx);
                // insert_cc_barrier();
                last_dest_out = !last_dest_out;
            }
            // insert_cc_barrier();
        }

        template <typename DataType, int local_size_x>
        void reduce_impl(SCBPtr<DataType> buf_in, SCBPtr<DataType> buf_out, uint32_t num_values, uint32_t wgx)
        {
            sequence.push_back([=](){
                auto ev = queue.submit([&](sycl::handler& cgh) {
                    DataType* in = buf_in->get_op_ptr();
                    DataType* out = buf_out->get_op_ptr();

                const auto num_subgroups = ceil_div<uint32_t>(local_size_x, engine->get_subgroup_size()); // This MUST divide evenly;
                const auto n = num_values;

                sycl::accessor <DataType, 1, sycl::access::mode::read_write, sycl::access::target::local>
                                        s(sycl::range<1>(num_subgroups), cgh);
                    // auto wait_list = queue.get_wait_list();
                    // cgh.depends_on(wait_list);
                    cgh.depends_on(barrier_events);
                    // maybe this can be rewritten in a simpler way?
                    cgh.parallel_for<class reduce_kernel>(
                        sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                        [=](auto it){
                            uint32_t idx = it.get_global_linear_id();
                            DataType sum = 0;
                            auto sg = it.get_sub_group();

                            // const uint32_t stride = gl_WorkGroupSize.x*gl_NumWorkGroups.x;
                            const uint32_t stride = it.get_group_range(0)*it.get_local_range(0);
                            while (idx < n) {
                                sum += in[idx];
                                idx += stride;
                            }

                            DataType total = sycl::reduce_over_group(sg, sum, sycl::plus<DataType>());

                            // double total = subgroupAdd(sum);


                            // if (subgroupElect()) {
                            if (sg.get_local_linear_id() == 0) {
                                s[sg.get_group_linear_id()] = total;
                            }

                            // group_barrier(it.get_group(), sycl::memory_scope::work_group);
                            group_barrier(it.get_group());
                            // it.barrier();
                            // memoryBarrierShared();
                            // barrier();

                            // if (gl_SubgroupID == 0) {
                            if (sg.get_group_linear_id() == 0) {
                                uint32_t j = sg.get_local_linear_id();//gl_SubgroupInvocationID;
                                DataType sum2 = 0;
                                while (j < num_subgroups) {
                                // while (j < gl_NumSubgroups) { // shared array size
                                    sum2 += s[j];
                                    // j += gl_SubgroupSize;
                                    j += sg.get_local_linear_range();
                                }
                                // combine and write
                                // double total2 = subgroupAdd(sum2);
                                DataType total2 = sycl::reduce_over_group(sg, sum2, sycl::plus<DataType>());
                                // if (subgroupElect()) {
                                if (sg.get_local_linear_id() == 0) {
                                    // data_out[gl_WorkGroupID.x] = total2;
                                    out[it.get_group_linear_id()] = total2;
                                }
                            }
                        });


                });
                                events.push_back(ev);

            });
        }


        template <typename DataType>
        void set_restore_lambda(uint32_t dim, SCBPtr<DataType> blocks, SCBPtr<DataType> backups,
                                SCBPtr<uint32_t> offsets, SCBPtr<DataType> lambda, uint32_t num_mat, uint32_t id_offset, uint32_t backup_offset, bool set)
        {
            sequence.push_back([=](){

                const uint32_t local_size_x = 256;
                const uint32_t items = num_mat * dim;
                const uint32_t wgx = ceil_div(items, local_size_x);
                // if (wgx * local_size_x < items)
                //     wgx++;

                DataType* previous = backups->get_op_ptr();
                DataType* values = blocks->get_op_ptr();
                DataType* buf_lambda = lambda->get_op_ptr();
                uint32_t* offset_list = offsets->get_op_ptr();
                // const DataType df = lambda->map()[0]; // get the damping factor - API should be redesigned so this can be avoided...
                const uint32_t rows = dim;
                constexpr bool backup = true;
                const bool set_diag = set;
                const uint32_t start_id = id_offset;
            

            auto ev = queue.submit([&](sycl::handler& cgh) {
                                    
                    cgh.parallel_for<class set_restore_lambda>(sycl::nd_range{sycl::range<1>{wgx*local_size_x}, sycl::range<1>{static_cast<size_t>(local_size_x)}}, 
                    [=](sycl::nd_item<1> it){
                            if (it.get_global_linear_id() < items) {
                            const uint32_t block_id = it.get_global_linear_id()/rows;
                            const uint32_t col = it.get_global_linear_id() % rows;
                            const uint32_t block_offset = offset_list[block_id+start_id] + rows*(col) + col;
                            const uint32_t boffset = it.get_global_linear_id() + backup_offset;
                            if (set_diag) {
                                if (backup) {
                                    previous[boffset] = values[block_offset];
                                }
                                values[block_offset] += buf_lambda[0];
                            }
                            else {
                                values[block_offset] = previous[boffset];
                            }
                        }

                    });

                });
                events.push_back(ev);

            });

        }

        void insert_wait() {
        //     // This is not great....
        //     sequence.push_back([this](){
        //         queue.wait();
        //     });
            

            // Specific to OpenSYCL

            // sequence.push_back([this](){
            //     // if (wait_list.size() > 0) {
            //         queue.submit([&](sycl::handler& cgh) {
            //         auto wait_list = queue.get_wait_list();
            //         cgh.depends_on(wait_list);
            //         // cgh.depends_on(events);
            //         events.clear();
            //         // cgh.memcpy(nullptr, nullptr, 0);
            //         cgh.copy((double*)nullptr, (double*)nullptr, 0);
            //         });
            //     // }
            // });
            // sequence.push_back([this](){
            //     barrierRequested = true;
            // });

            sequence.push_back([this]() {
                barrier_events = queue.get_wait_list();
            });
        }

        // void process_barrier(sycl::queue & queue, sycl::handler& cgh) {
        //     if (barrierRequested) {
        //         cgh.depends_on(queue.get_wait_list());
        //         barrierRequested = false;
        //     }
        // }

    public:
        void execute()
        {

            if (init && init_sequence.size() > 0) {
                for (auto && f: init_sequence) {
                    f();
                }
                queue.wait();
                init = false;
                // std::cout << "did init!\n";
                // std::cout << "is_gpu: " << queue.get_device().is_gpu() << std::endl;
            }
            if (sequence.size() > 0) {
                for (auto && f: sequence) {
                    f();
                }
                queue.wait();
            }

        }
    };

}