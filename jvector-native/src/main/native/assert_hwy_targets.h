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

#if defined(__x86_64__) || defined(_M_X64)
// AVX3_DL and AVX3_SPR tiers cannot use their own fine-grained march under
// MSVC (no icelake-server / sapphirerapids target).  Both are compiled with
// /arch:AVX512, so Highway selects HWY_AVX3 — assert that here instead.
#if defined(JV_REQUIRE_HWY_AVX3_SPR)
#  ifdef _MSC_VER
#    if HWY_STATIC_TARGET != HWY_AVX3
#      error "MSVC AVX3_SPR build: expected Highway to select HWY_AVX3 (/arch:AVX512). Check compiler flags."
#    endif
#  else
#    if HWY_STATIC_TARGET != HWY_AVX3_SPR
#      error "Highway did not select HWY_AVX3_SPR for the Sapphire Rapids build. Check compiler flags, compiler support, and Highway blocklists."
#    endif
#  endif
#elif defined(JV_REQUIRE_HWY_AVX3_DL)
#  ifdef _MSC_VER
#    if HWY_STATIC_TARGET != HWY_AVX3
#      error "MSVC AVX3_DL build: expected Highway to select HWY_AVX3 (/arch:AVX512). Check compiler flags."
#    endif
#  else
#    if HWY_STATIC_TARGET != HWY_AVX3_DL
#      error "Highway did not select HWY_AVX3_DL for the Ice Lake build. Check compiler flags, compiler support, and Highway blocklists."
#    endif
#  endif
#elif defined(JV_REQUIRE_HWY_AVX3)
#  if HWY_STATIC_TARGET != HWY_AVX3
#    error "Highway did not select HWY_AVX3 for the AVX-512 build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#elif defined(JV_REQUIRE_HWY_AVX2)
#  if HWY_STATIC_TARGET != HWY_AVX2
#    error "Highway did not select HWY_AVX2 for the AVX2 build. Check compiler flags, compiler support, and Highway blocklists."
#  endif
#endif
#endif // __X86_64__
