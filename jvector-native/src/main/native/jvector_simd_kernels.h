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

// Header file for the SIMD kernels
// Kernel declarations are auto-generated from jvector_simd_kernel_list.h
#ifndef SIMD_KERNELS_H
#define SIMD_KERNELS_H

#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// Portable optimisation hints
//
// JV_OPTIMIZE(opts)
//   GCC/Clang: __attribute__((optimize(opts))) — request extra optimisation
//   passes on a single function (e.g. "rename-registers").
//   MSVC: no equivalent; expands to nothing.
//
// JV_PRAGMA_UNROLL(N)
//   GCC/Clang: emits "#pragma GCC unroll N" via _Pragma, hinting the compiler
//   to unroll the immediately following loop N times.
//   MSVC: no equivalent; expands to nothing.
//   Usage: place on its own line directly before the for/while statement.
// ---------------------------------------------------------------------------
#if defined(_MSC_VER)
#  define JV_OPTIMIZE(opts)
#  define JV_PRAGMA_UNROLL(N)
#else
#  define JV_OPTIMIZE(opts)       __attribute__((optimize(opts)))
#  define JV_PRAGMA_UNROLL(N)     _Pragma(JV_STRINGIFY_(GCC unroll N))
#  define JV_STRINGIFY_(x)        JV_STRINGIFY2_(x)
#  define JV_STRINGIFY2_(x)       #x
#endif

// Macro to declare a kernel function signature from the kernel list
#define KERNEL_ENTRY(ret_type, name, params, names) \
    ret_type name params;

// Generate namespace declarations for each ISA
#define DECLARE_SIMD_KERNELS(ISA) \
    namespace ISA { \
    JVECTOR_SIMD_KERNEL_LIST \
    }

#include "jvector_simd_kernel_list.h"

DECLARE_SIMD_KERNELS(AVX3_SPR)
DECLARE_SIMD_KERNELS(AVX3_DL)
DECLARE_SIMD_KERNELS(AVX3)
DECLARE_SIMD_KERNELS(AVX2)
DECLARE_SIMD_KERNELS(SSE42)

#undef KERNEL_ENTRY

#endif // SIMD_KERNELS_H

