/*
 * Copyright DataStax, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Sapphire Rapids-only kernels compiled with -march=sapphirerapids.
// This file contains only kernels that require AVX-512 FP16 instructions and
// therefore cannot share the jvector_simd_kernels.cpp compilation unit.
// All other kernels for the AVX3_SPR vtable are reused directly from AVX3::.

#include <cstdint>
#include "jvector_simd.h"
#include "hwy/highway.h"
#include "assert_hwy_targets.h"

namespace hn = hwy::HWY_NAMESPACE;

// =============================================================================
// FP16 dot-product — native AVX-512 FP16 path
// =============================================================================
//
// ScalableTag<float16_t> gives 32 lanes on a 512-bit register.
// MulAdd maps to _mm512_fmadd_ph.  The 4x unrolled loop hides FMA latency
// identically to the float DotProductImpl in jvector_simd_kernels.cpp.
// ReduceSum on float16_t promotes through float32 internally (generic_ops-inl.h)
// and returns float16_t; static_cast<float> gives the public API result type.

template <class Tag>
HWY_INLINE float DotProductF16Impl(Tag tag,
                                    const hwy::float16_t *a,
                                    const hwy::float16_t *b,
                                    size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto acc0 = hn::Zero(tag), acc1 = hn::Zero(tag);
    auto acc2 = hn::Zero(tag), acc3 = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        acc0 = hn::MulAdd(hn::LoadU(tag, a + ii + 0*lanes), hn::LoadU(tag, b + ii + 0*lanes), acc0);
        acc1 = hn::MulAdd(hn::LoadU(tag, a + ii + 1*lanes), hn::LoadU(tag, b + ii + 1*lanes), acc1);
        acc2 = hn::MulAdd(hn::LoadU(tag, a + ii + 2*lanes), hn::LoadU(tag, b + ii + 2*lanes), acc2);
        acc3 = hn::MulAdd(hn::LoadU(tag, a + ii + 3*lanes), hn::LoadU(tag, b + ii + 3*lanes), acc3);
    }
    auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
    for (; ii + lanes <= size; ii += lanes) {
        acc = hn::MulAdd(hn::LoadU(tag, a + ii), hn::LoadU(tag, b + ii), acc);
    }
    if (ii < size) {
        acc = hn::MulAdd(hn::LoadN(tag, a + ii, size - ii),
                         hn::LoadN(tag, b + ii, size - ii), acc);
    }
    return static_cast<float>(hn::ReduceSum(tag, acc));
}

namespace AVX3_SPR {

HWY_FLATTEN float dot_product_f16(
        const uint16_t *a, size_t aoffset, const uint16_t *b, size_t boffset, size_t length)
{
    // uint16_t and hwy::float16_t have identical 2-byte storage; reinterpret.
    const auto *af16 = reinterpret_cast<const hwy::float16_t *>(a + aoffset);
    const auto *bf16 = reinterpret_cast<const hwy::float16_t *>(b + boffset);
    return DotProductF16Impl(hn::ScalableTag<hwy::float16_t>{}, af16, bf16, length);
}

} // namespace AVX3_SPR
