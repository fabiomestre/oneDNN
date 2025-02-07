/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
* Copyright 2020 Codeplay Software Limited
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "gpu/nvidia/cudnn_matmul.hpp"

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"

#include "gpu/nvidia/cudnn_matmul_executor.hpp"
#include "gpu/nvidia/sycl_cuda_engine.hpp"
#include "gpu/nvidia/sycl_cuda_scoped_context.hpp"
#include "gpu/nvidia/sycl_cuda_stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace nvidia {

status_t cudnn_matmul_t::execute(const exec_ctx_t &ctx) const {
    const bool with_bias = matmul_impl_->with_bias();
    const bool has_runtime_args = matmul_impl_->has_runtime_params();

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bias_d = with_bias
            ? ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1))
            : nullptr;

    status_t status;
    if (has_runtime_args) {
        // Initialise all runtime parameters
        status = matmul_impl_->init_parameters(src_d, weights_d, dst_d, bias_d);
        if (status != status::success) return status;
    }

    nvidia::sycl_cuda_stream_t *cuda_stream
            = utils::downcast<nvidia::sycl_cuda_stream_t *>(ctx.stream());

    if (!pd()->attr()->output_scales_.defined()) {
        cuda_stream->interop_task([&](::sycl::handler &cgh) {
            auto arg_out_scales
                    = CTX_IN_SYCL_MEMORY(DNNL_ARG_ATTR_OUTPUT_SCALES);

            compat::host_task(cgh, [=](const compat::interop_handle &ih) {
                auto &sycl_engine = *utils::downcast<sycl_cuda_engine_t *>(
                        cuda_stream->engine());
                auto sc = cuda_sycl_scoped_context_handler_t(sycl_engine);

                uint8_t *out_scale_ptr = static_cast<uint8_t *>(
                        arg_out_scales.get_native_pointer(ih));
                uint8_t *out_scale_dst_ptr
                        = reinterpret_cast<uint8_t *>(output_scale_);

                // output_scale only supports single value floats
                size_t out_scale_size = sizeof(float);
                CUDA_EXECUTE_FUNC(cuMemcpyAsync, (CUdeviceptr)out_scale_dst_ptr,
                        (CUdeviceptr)out_scale_ptr, out_scale_size,
                        cuda_stream->get_underlying_stream());
                cudaDeviceSynchronize();
            });
        });
    }

    const auto scratchpad_type = matmul_impl_->get_scratchpad_type();
    const auto scratchpad_size = matmul_impl_->with_scratchpad()
            ? (dst_d.nelems() * types::data_type_size(scratchpad_type))
            : 0;

    status = executor_->execute(ctx, ctx.stream()->engine(), matmul_impl_,
            output_scale_, scratchpad_size);

    if (has_runtime_args) {
        auto &evts = cuda_stream->get_deps();
        for (auto e : evts) {
            e.wait();
        }

        matmul_impl_->cleanup();
    }

    return status;
}

} // namespace nvidia
} // namespace gpu
} // namespace impl
} // namespace dnnl
