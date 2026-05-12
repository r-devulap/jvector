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

#include <inttypes.h>
#include <float.h>
#include <math.h>
#include <cstring>
#include "jvector_simd.h"
#include "hwy/highway.h"

// =============================================================================
// Highway macro usage in this file
// =============================================================================
//
// HWY_INLINE
//   Expands to `inline __attribute__((always_inline))` on GCC/Clang.
//   Used on every helper that participates in a hot SIMD loop to prevent the
//   compiler from ever emitting a real call and to keep register pressure
//   visible across the inlined body.  Prefer this over plain `inline` for any
//   function that contains SIMD intrinsics.
//
// HWY_FLATTEN
//   Expands to `__attribute__((flatten))`, which asks the compiler to inline
//   *all* callees into the annotated function.  Used on the public entry-points
//   (assemble_and_sum_f32_512, pq_decoded_cosine_similarity_f32_512, the three
//   distance wrappers) so that the multi-target Highway dispatch stub sees a
//   single monolithic body with no residual call overhead.
//
// HWY_RESTRICT
//   Portable spelling of `__restrict__` / `restrict`.  Tells the compiler that
//   a pointer does not alias any other pointer in scope, so loads through it
//   remain valid across stores made through a different pointer.  Applied to:
//     - Load helper parameters (e.g. LoadDup256) — ensures the loaded value is
//       treated as loop-invariant even when the caller stores to an accumulator.
//     - calculate_partial_sums_f32 inputs (codebook, query) and output
//       (partialSums) — prevents the compiler from reloading read-only inputs
//       after each write to partialSums.
//   Not needed when inputs and output already have different types (e.g.
//   float* vs unsigned char*), because C++ strict-aliasing rules already
//   guarantee they cannot alias.
//
// =============================================================================
// Highway API tutorial — intrinsics used in this file
// =============================================================================
//
// --- Tags (describe vector type and width) ---
//
// ScalableTag<T>
//   Represents the full native SIMD width for type T.
//   e.g. ScalableTag<float> is 8 lanes on AVX2, 16 lanes on AVX-512.
//   Used in the main loop bodies where we want the widest available vector.
//
// CappedTag<T, N>  /  HWY_CAPPED(T, N)
//   A tag capped to at most N lanes, even on wider ISAs.
//   Used in the small-vector fast paths (e.g. size==4, size==8) so that we
//   avoid wasting the extra lanes of a wide register on tiny inputs.
//
// Half<D>
//   Produces a tag whose lane count is half that of D.
//   Used in LoadDup256 to load 8 floats into the lower half of a 512-bit
//   register before Combine duplicates them into the upper half.
//
// Rebind<NewT, Tag>  /  RebindToSigned<Tag>
//   Produce a new tag of the same width but a different element type.
//   Used in assemble_and_sum_f32_512 and pq_decoded_cosine_similarity to
//   reinterpret the float-width register as uint8/uint16/int32 during the
//   index promotion pipeline.
//
// Lanes(tag)   — runtime lane count for the given tag.
// MaxLanes(tag) — compile-time upper bound on lane count (used in static_assert
//                 and constexpr branches).
//
// --- Vector type ---
//
// Vec<Tag>
//   The SIMD vector type corresponding to a tag.
//   All arithmetic and load/store operations return or accept Vec<Tag>.
//
// --- Initialisation ---
//
// Zero(tag)         — vector of all zeros; used to initialise accumulators.
// Set(tag, scalar)  — broadcast a scalar to every lane.
// Iota(tag, start)  — fill lanes with start, start+1, start+2, …
//                     Used to build the running index vector for GatherIndex.
//
// --- Loads ---
//
// LoadU(tag, ptr)        — unaligned load of Lanes(tag) elements from ptr.
// LoadN(tag, ptr, n)     — load n elements; remaining lanes are zero-padded.
//                          Used for loop tails without a branch per element.
// LoadDup128(tag, ptr)   — load 128 bits and broadcast across the full vector.
//                          Used for size==2 and size==4 query vectors so the
//                          same query chunk lines up with every centroid chunk.
// MaskedLoad(mask, tag, ptr)
//                        — load only the lanes where mask is set; others zero.
//                          Used in CosineDistance tail handling.
//
// --- Store ---
//
// StoreU(vec, tag, ptr)  — unaligned store of Lanes(tag) elements to ptr.
//
// --- Arithmetic ---
//
// Add(a, b)              — lane-wise addition.
// Sub(a, b)              — lane-wise subtraction.
// Mul(a, b)              — lane-wise multiplication.
// MulAdd(a, b, c)        — fused multiply-add: (a * b) + c.
//                          Preferred over separate Mul+Add for FMA throughput.
//
// --- Type promotion ---
//
// PromoteTo(narrower_tag, vec)
//   Zero-extends each element to the wider type.
//   Used twice in the gather pipeline: u8 → u16 → i32, so that byte offsets
//   become 32-bit gather indices without sign-extension artefacts.
//
// --- Gather ---
//
// GatherIndex(tag, base_ptr, index_vec)
//   Loads one element per lane using per-lane 32-bit indices (in elements,
//   not bytes).  Used to collect PQ lookup-table entries and codebook floats
//   whose positions are determined at runtime by the encoded offsets.
//
// --- Reductions ---
//
// ReduceSum(tag, vec)    — horizontal sum of all lanes; returns a scalar.
//
// --- Horizontal-reduction shuffles (used in calculate_partial_sums_f32) ---
//
// Shuffle2301(vec)
//   Permutes pairs of adjacent lanes: [0,1,2,3] → [2,3,0,1].
//   Adding a vector with its Shuffle2301 partner sums adjacent pairs.
//
// Shuffle1032(vec)
//   Permutes 32-bit elements within each 128-bit lane: [0,1,2,3] → [1,0,3,2].
//   Used as a second reduction step to sum the results of Shuffle2301.
//
// SwapAdjacentBlocks(vec)
//   Swaps the two 128-bit halves of each 256-bit block.
//   On AVX-512 (512-bit vector = four 128-bit blocks) this is used as the
//   first step of the size==8 horizontal reduction before Shuffle1032/2301.
//
// --- Masks ---
//
// FirstN(tag, n)
//   Returns a mask with the first n lanes set and the rest clear.
//   Used with MaskedLoad to handle the tail of a vector that doesn't fill
//   a full register.
//
// --- Combine ---
//
// Combine(d, hi, lo)
//   Concatenates two half-width vectors into one full-width vector.
//   Used in LoadDup256 to duplicate 8 floats across both halves of a
//   512-bit register (lo = hi = the same 256-bit load).
//

namespace hn = hwy::HWY_NAMESPACE;

// Loads 8 floats from ptr and broadcasts them to fill the full vector D.
// On ISAs where D is exactly 8 lanes (e.g. AVX2) this is a plain LoadU.
// On wider ISAs (e.g. AVX-512, 16 lanes) the 8 floats are loaded into the
// 256-bit half-tag and then Combine'd to duplicate them into both halves.
// NOTE: not designed for ISAs wider than 512-bit (would need additional Combine levels).
// HWY_RESTRICT tells the compiler that ptr does not alias any accumulator or
// other pointer visible at the call site, allowing it to treat all loads from
// ptr as invariant across iterations and hoist them freely.
template <class D>
HWY_INLINE hn::Vec<D> LoadDup256(D d, const float *HWY_RESTRICT ptr)
{
    static_assert(hn::MaxLanes(d) <= 16,
                  "LoadDup256 is not implemented for ISAs wider than 512-bit");
    if constexpr (hn::MaxLanes(d) > 8) {
        const hn::Half<D> dh;
        const auto half = hn::LoadU(dh, ptr);
        return hn::Combine(d, half, half);
    }
    else {
        return hn::LoadU(d, ptr);
    }
}
// =============================================================================
// Base Fp32 kernels
// =============================================================================

// Dot product kernel templated on the Highway tag type, shared by the
// full-width ScalableTag path and the HWY_CAPPED fast paths.
// The 4x unrolled loop hides FMA latency; for capped tags the loop body
// is never entered for small sizes and the single-vector path handles them.
template <class Tag>
HWY_INLINE float DotProductImpl(Tag tag, const float *a, const float *b, size_t size)
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
    return hn::ReduceSum(tag, acc);
}

// L2 square distance kernel templated on the Highway tag type, shared by
// the full-width ScalableTag path and the HWY_CAPPED fast paths.
template <class Tag>
HWY_INLINE float L2SquareDistanceImpl(Tag tag, const float *a, const float *b, size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto acc0 = hn::Zero(tag), acc1 = hn::Zero(tag);
    auto acc2 = hn::Zero(tag), acc3 = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + 4 * lanes <= size; ii += 4 * lanes) {
        auto d0 = hn::LoadU(tag, a + ii + 0*lanes) - hn::LoadU(tag, b + ii + 0*lanes);
        auto d1 = hn::LoadU(tag, a + ii + 1*lanes) - hn::LoadU(tag, b + ii + 1*lanes);
        auto d2 = hn::LoadU(tag, a + ii + 2*lanes) - hn::LoadU(tag, b + ii + 2*lanes);
        auto d3 = hn::LoadU(tag, a + ii + 3*lanes) - hn::LoadU(tag, b + ii + 3*lanes);
        acc0 = hn::MulAdd(d0, d0, acc0);
        acc1 = hn::MulAdd(d1, d1, acc1);
        acc2 = hn::MulAdd(d2, d2, acc2);
        acc3 = hn::MulAdd(d3, d3, acc3);
    }
    auto acc = hn::Add(hn::Add(acc0, acc1), hn::Add(acc2, acc3));
    for (; ii + lanes <= size; ii += lanes) {
        auto d = hn::LoadU(tag, a + ii) - hn::LoadU(tag, b + ii);
        acc = hn::MulAdd(d, d, acc);
    }
    if (ii < size) {
        auto d = hn::LoadN(tag, a + ii, size - ii) - hn::LoadN(tag, b + ii, size - ii);
        acc = hn::MulAdd(d, d, acc);
    }
    return hn::ReduceSum(tag, acc);
}

// Cosine distance kernel templated on the Highway tag type, shared by
// the full-width ScalableTag path and the HWY_CAPPED fast paths.
template <class Tag>
HWY_INLINE float CosineDistanceImpl(Tag tag, const float *a, const float *b, size_t size)
{
    const size_t lanes = hn::Lanes(tag);
    auto sum_ab = hn::Zero(tag), sum_aa = hn::Zero(tag), sum_bb = hn::Zero(tag);
    size_t ii = 0;
    for (; ii + lanes <= size; ii += lanes) {
        auto va = hn::LoadU(tag, a + ii);
        auto vb = hn::LoadU(tag, b + ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    if (ii < size) {
        auto va = hn::LoadN(tag, a + ii, size - ii);
        auto vb = hn::LoadN(tag, b + ii, size - ii);
        sum_ab = hn::MulAdd(va, vb, sum_ab);
        sum_aa = hn::MulAdd(va, va, sum_aa);
        sum_bb = hn::MulAdd(vb, vb, sum_bb);
    }
    return hn::ReduceSum(tag, sum_ab)
           / sqrtf(hn::ReduceSum(tag, sum_aa) * hn::ReduceSum(tag, sum_bb));
}

// Returns the dot product sum(a[ii] * b[ii]).
//
// Short-vector fast paths: when the register width is wider than the vector
// (e.g. a 4-element input on AVX-512), using the full register wastes lanes
// and can hurt latency.  Capped tags keep execution in narrow registers.
HWY_INLINE float DotProduct(const float *a,
                            size_t aoffset,
                            const float *b,
                            size_t boffset,
                            size_t length)
{
    a += aoffset;
    b += boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return DotProductImpl(HWY_CAPPED(float, 4){}, a, b, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return DotProductImpl(HWY_CAPPED(float, 8){}, a, b, length); }
#endif
#endif
    return DotProductImpl(hn::ScalableTag<float>{}, a, b, length);
}

HWY_INLINE float CosineDistance(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    const float *ap = a + aoffset;
    const float *bp = b + boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return CosineDistanceImpl(HWY_CAPPED(float, 4){}, ap, bp, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return CosineDistanceImpl(HWY_CAPPED(float, 8){}, ap, bp, length); }
#endif
#endif
    return CosineDistanceImpl(hn::ScalableTag<float>{}, ap, bp, length);
}

HWY_INLINE float L2SquareDistance(const float *a,
                                  size_t aoffset,
                                  const float *b,
                                  size_t boffset,
                                  size_t length)
{
    a += aoffset;
    b += boffset;
#if HWY_MAX_BYTES > 16
    if (length <= 4) { return L2SquareDistanceImpl(HWY_CAPPED(float, 4){}, a, b, length); }
#if HWY_MAX_BYTES > 32
    if (length <= 8) { return L2SquareDistanceImpl(HWY_CAPPED(float, 8){}, a, b, length); }
#endif
#endif
    return L2SquareDistanceImpl(hn::ScalableTag<float>{}, a, b, length);
}

HWY_FLATTEN float cosine_f32_512_native(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return CosineDistance(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float dot_product_f32_512_native(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return DotProduct(a, aoffset, b, boffset, length);
}

HWY_FLATTEN float euclidean_f32_512_native(
        const float *a, size_t aoffset, const float *b, size_t boffset, size_t length)
{
    return L2SquareDistance(a, aoffset, b, boffset, length);
}

// =============================================================================
// Element-wise in-place arithmetic and reduction kernels
// =============================================================================
//
// rename-registers: extra GCC register-renaming pass that breaks false WAR/WAW
//   hazards between short-lived zmm values, enabling more ILP in the SIMD loops.
// #pragma GCC unroll 4: unroll by 4 to hide the 4-cycle FMA latency and keep
//   both AVX-512 FMA ports saturated across independent load–op–store chains.
//
__attribute__((optimize("rename-registers")))
HWY_FLATTEN void add_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Add(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Add(a, b), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void add_scalar_in_place_f32(float *HWY_RESTRICT v1,
                                          float value,
                                          size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto vval = hn::Set(d, value);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        hn::StoreU(hn::Add(a, vval), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        hn::StoreN(hn::Add(a, vval), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void sub_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Sub(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Sub(a, b), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void sub_scalar_in_place_f32(float *HWY_RESTRICT v1,
                                          float value,
                                          size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    const auto vval = hn::Set(d, value);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        hn::StoreU(hn::Sub(a, vval), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        hn::StoreN(hn::Sub(a, vval), d, v1 + i, rem);
    }
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN float max_f32(const float *HWY_RESTRICT v, size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    auto accum = hn::Set(d, -FLT_MAX);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        accum = hn::Max(accum, hn::LoadU(d, v + i));
    }
    float result = hn::ReduceMax(d, accum);
    for (; i < length; i++) {
        if (v[i] > result) result = v[i];
    }
    return result;
}

__attribute__((optimize("rename-registers")))
HWY_FLATTEN void min_in_place_f32(float *HWY_RESTRICT v1,
                                   const float *HWY_RESTRICT v2,
                                   size_t length)
{
    hn::ScalableTag<float> d;
    const size_t lanes = hn::Lanes(d);
    size_t i = 0;
#pragma GCC unroll 4
    for (; i + lanes <= length; i += lanes) {
        auto a = hn::LoadU(d, v1 + i);
        auto b = hn::LoadU(d, v2 + i);
        hn::StoreU(hn::Min(a, b), d, v1 + i);
    }
    if (i < length) {
        const size_t rem = length - i;
        auto a = hn::LoadN(d, v1 + i, rem);
        auto b = hn::LoadN(d, v2 + i, rem);
        hn::StoreN(hn::Min(a, b), d, v1 + i, rem);
    }
}

