//=== signbit.hpp -   Unary function signbit            ------  *-C++-*--/===//
//
//                      Data Parallel Control (dpctl)
//
// Copyright 2020-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===---------------------------------------------------------------------===//
///
/// \file
/// This file defines kernels for elementwise evaluation of SIGNBIT(x)
/// function that tests whether the sign bit of the tensor element is set.
//===---------------------------------------------------------------------===//

#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "vec_size_util.hpp"

#include "kernels/dpctl_tensor_types.hpp"

#include "utils/offset_utils.hpp"
#include "utils/type_dispatch_building.hpp"
#include "utils/type_utils.hpp"

namespace dpctl
{
namespace tensor
{
namespace kernels
{
namespace signbit
{

namespace td_ns = dpctl::tensor::type_dispatch;

using dpctl::tensor::type_utils::is_complex;
using dpctl::tensor::type_utils::vec_cast;

template <typename argT, typename resT> struct SignbitFunctor
{
    static_assert(std::is_same_v<resT, bool>);

    using is_constant = std::false_type;
    static constexpr resT constant_value = false;
    using supports_vec = std::true_type;
    using supports_sg_loadstore = std::true_type;

    resT operator()(const argT &in) const { return std::signbit(in); }

    template <int vec_sz>
    sycl::vec<resT, vec_sz> operator()(const sycl::vec<argT, vec_sz> &in) const
    {
        auto const &res_vec = sycl::signbit(in);

        using deducedT = typename std::remove_cv_t<
            std::remove_reference_t<decltype(res_vec)>>::element_type;

        return vec_cast<resT, deducedT, vec_sz>(res_vec);
    }
};

template <typename argT,
          typename resT = bool,
          std::uint8_t vec_sz = 4u,
          std::uint8_t n_vecs = 2u,
          bool enable_sg_loadstore = true>
using SignbitContigFunctor =
    elementwise_common::UnaryContigFunctor<argT,
                                           resT,
                                           SignbitFunctor<argT, resT>,
                                           vec_sz,
                                           n_vecs,
                                           enable_sg_loadstore>;

template <typename argTy, typename resTy, typename IndexerT>
using SignbitStridedFunctor = elementwise_common::
    UnaryStridedFunctor<argTy, resTy, IndexerT, SignbitFunctor<argTy, resTy>>;

template <typename argTy> struct SignbitOutputType
{
    using value_type = typename std::disjunction<
        td_ns::TypeMapResultEntry<argTy, sycl::half, bool>,
        td_ns::TypeMapResultEntry<argTy, float, bool>,
        td_ns::TypeMapResultEntry<argTy, double, bool>,
        td_ns::DefaultResultEntry<void>>::result_type;

    static constexpr bool is_defined = !std::is_same_v<value_type, void>;
};

namespace
{

namespace vsu_ns = dpctl::tensor::kernels::vec_size_utils;

using vsu_ns::ContigHyperparameterSetDefault;
using vsu_ns::UnaryContigHyperparameterSetEntry;

template <typename argTy> struct SignbitContigHyperparameterSet
{
    using value_type =
        typename std::disjunction<ContigHyperparameterSetDefault<4u, 2u>>;

    constexpr static auto vec_sz = value_type::vec_sz;
    constexpr static auto n_vecs = value_type::n_vecs;
};

} // end of anonymous namespace

template <typename T1, typename T2, std::uint8_t vec_sz, std::uint8_t n_vecs>
class signbit_contig_kernel;

template <typename argTy>
sycl::event signbit_contig_impl(sycl::queue &exec_q,
                                size_t nelems,
                                const char *arg_p,
                                char *res_p,
                                const std::vector<sycl::event> &depends = {})
{
    constexpr std::uint8_t vec_sz =
        SignbitContigHyperparameterSet<argTy>::vec_sz;
    constexpr std::uint8_t n_vecs =
        SignbitContigHyperparameterSet<argTy>::n_vecs;

    return elementwise_common::unary_contig_impl<
        argTy, SignbitOutputType, SignbitContigFunctor, signbit_contig_kernel,
        vec_sz, n_vecs>(exec_q, nelems, arg_p, res_p, depends);
}

template <typename fnT, typename T> struct SignbitContigFactory
{
    fnT get()
    {
        if constexpr (!SignbitOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = signbit_contig_impl<T>;
            return fn;
        }
    }
};

template <typename fnT, typename T> struct SignbitTypeMapFactory
{
    /*! @brief get typeid for output type of sycl::isinf(T x) */
    std::enable_if_t<std::is_same<fnT, int>::value, int> get()
    {
        using rT = typename SignbitOutputType<T>::value_type;
        return td_ns::GetTypeid<rT>{}.get();
    }
};

template <typename T1, typename T2, typename T3> class signbit_strided_kernel;

template <typename argTy>
sycl::event
signbit_strided_impl(sycl::queue &exec_q,
                     size_t nelems,
                     int nd,
                     const ssize_t *shape_and_strides,
                     const char *arg_p,
                     ssize_t arg_offset,
                     char *res_p,
                     ssize_t res_offset,
                     const std::vector<sycl::event> &depends,
                     const std::vector<sycl::event> &additional_depends)
{
    return elementwise_common::unary_strided_impl<argTy, SignbitOutputType,
                                                  SignbitStridedFunctor,
                                                  signbit_strided_kernel>(
        exec_q, nelems, nd, shape_and_strides, arg_p, arg_offset, res_p,
        res_offset, depends, additional_depends);
}

template <typename fnT, typename T> struct SignbitStridedFactory
{
    fnT get()
    {
        if constexpr (!SignbitOutputType<T>::is_defined) {
            fnT fn = nullptr;
            return fn;
        }
        else {
            fnT fn = signbit_strided_impl<T>;
            return fn;
        }
    }
};

} // namespace signbit
} // namespace kernels
} // namespace tensor
} // namespace dpctl
