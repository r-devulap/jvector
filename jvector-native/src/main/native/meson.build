# Copyright DataStax, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

project('jvector_simd_kernels', 'cpp',
  version: '0.1.0',
  meson_version: '>= 0.55.0',
  default_options: ['cpp_std=c++17']
)

# Highway headers are used as an include directory only.
# HWY_COMPILE_ONLY_STATIC / HWY_COMPILE_ONLY_SCALAR bypass the
# dynamic-dispatch runtime, so only headers are needed at compile time.
hwy_inc = include_directories('third_party/highway')

cc = meson.get_compiler('cpp')
is_windows = host_machine.system() == 'windows'
is_msvc = cc.get_argument_syntax() == 'msvc'

# -fvisibility=hidden is a GCC/Clang flag that hides all symbols by default so
# that only explicitly annotated symbols end up in the shared-library ABI.
# MSVC controls DLL symbol visibility via __declspec(dllexport/dllimport) in the
# source (see jvector_simd.h), so the flag is omitted there.
visibility_flag = is_msvc ? [] : ['-fvisibility=hidden']

# ---------------------------------------------------------------------------
# Per-ISA compiler flags
#
# GCC/Clang: use fine-grained -march= targets so the compiler emits only the
#   instruction extensions belonging to each tier.  -maes/-mpclmul enable the
#   corresponding intrinsics on top of the base -march.
#
# MSVC: /arch:AVX512 is the only AVX-512 flag available; MSVC does not expose
#   microarchitecture-level targets like icelake-server or sapphirerapids.
#   Highway's MSVC target detection will therefore select HWY_AVX3 for any
#   AVX-512 compilation unit.  Because of this, AVX3_DL and AVX3_SPR tiers
#   cannot be compiled under MSVC and are skipped entirely (see below).
#   For AVX2, /arch:AVX2 is the correct flag.
#   For the SSE4.2 baseline, no /arch: flag is needed — SSE4.2 is always
#   available on x86-64 under MSVC.
# ---------------------------------------------------------------------------

# Arch flags per tier — one variable each so the flag string is never repeated.
if is_msvc
  # Highway's detect_targets.h marks HWY_AVX3 and below as broken under MSVC
  # due to historic codegen bugs (HWY_BROKEN_MSVC).  We override that to 0 so
  # that /arch:AVX512 correctly selects HWY_AVX3.  The tradeoff is accepting
  # MSVC's AVX-512 codegen quality, which is sufficient for correctness.
  hwy_broken_msvc_override = ['-DHWY_BROKEN_MSVC=0']
  avx3_flag          = ['/arch:AVX512']
  avx3_dl_flag       = ['/arch:AVX512']   # best available under MSVC
  avx3_spr_flag      = ['/arch:AVX512']   # best available under MSVC
  avx2_flag          = ['/arch:AVX2']
  sse42_flag         = []                 # SSE4.2 always present on MSVC x86-64
else
  hwy_broken_msvc_override = []
  avx3_flag          = ['-march=skylake-avx512']
  avx3_dl_flag       = ['-march=icelake-server']
  avx3_spr_flag      = ['-march=sapphirerapids']
  avx2_flag          = ['-march=haswell', '-maes']
  sse42_flag         = ['-msse4.2', '-mpclmul', '-maes']
endif

isa_variants = [
  {
    'name'     : 'avx3',
    'namespace': 'AVX3',
    'args'     : avx3_flag + hwy_broken_msvc_override + ['-DHWY_COMPILE_ONLY_STATIC',
                               '-DJV_REQUIRE_HWY_AVX3'],
  },
  {
    'name'     : 'avx2',
    'namespace': 'AVX2',
    'args'     : avx2_flag + hwy_broken_msvc_override + ['-DHWY_COMPILE_ONLY_STATIC',
                               '-DJV_REQUIRE_HWY_AVX2'],
  },
  {
    'name'     : 'sse42',
    'namespace': 'SSE42',
    'args'     : sse42_flag + hwy_broken_msvc_override + ['-DHWY_COMPILE_ONLY_STATIC',
                                '-DJV_REQUIRE_HWY_SCALAR'],
  },
]

isa_libs = []
foreach isa : isa_variants
  lib = static_library(
    'simdKernels_' + isa['name'],
    sources            : 'jvector_simd_kernels.cpp',
    include_directories: hwy_inc,
    cpp_args           : isa['args'] + ['-DJV_ISA=' + isa['namespace']] + visibility_flag
  )
  isa_libs += lib
endforeach

# AVX3_DL (Ice Lake) tier: -march=icelake-server enables the full ICX feature
# set (AVX3 + VNNI, VBMI, VBMI2, IFMA, BITALG, VPOPCNTDQ, GFNI, VAES, VPCLMULQDQ).
# Under MSVC, avx3_dl_flag is /arch:AVX512 (the best available); Highway will
# select HWY_AVX3 rather than HWY_AVX3_DL, which is asserted in
# assert_hwy_targets.h.  When ICX-specific kernels are added to this file that
# require intrinsics unavailable under /arch:AVX512, guard them with #ifndef _MSC_VER.
avx3_dl_lib = static_library(
  'simdKernels_avx3_dl',
  sources            : 'jvector_avx3_dl_kernels.cpp',
  include_directories: hwy_inc,
  cpp_args           : avx3_dl_flag + hwy_broken_msvc_override + [
    '-DHWY_COMPILE_ONLY_STATIC',
    '-DJV_REQUIRE_HWY_AVX3_DL',
  ] + visibility_flag,
)
isa_libs += avx3_dl_lib

# AVX3_SPR (Sapphire Rapids) tier: -march=sapphirerapids adds AVX512FP16 and
# AVX512BF16 on top of the icelake-server feature set.
# Requires GCC >= 12 or Clang >= 14.
# Under MSVC, avx3_spr_flag is /arch:AVX512 for the same reasons as AVX3_DL.
avx3_spr_lib = static_library(
  'simdKernels_avx3_spr',
  sources            : 'jvector_avx3_spr_kernels.cpp',
  include_directories: hwy_inc,
  cpp_args           : avx3_spr_flag + hwy_broken_msvc_override + [
    '-DHWY_COMPILE_ONLY_STATIC',
    '-DJV_REQUIRE_HWY_AVX3_SPR',
  ] + visibility_flag,
)
isa_libs += avx3_spr_lib

# vectorUtil provides a single runtime-dispatch entry point.
# link_whole pulls every ISA object into the shared library so that no
# symbols are dropped when consumers link against it.
# abort.cc defines hwy::Abort, referenced by the SSE4 scalar fallbacks in
# generic_ops-inl.h. It was extracted from targets.cc specifically for
# projects that manage their own dispatch (google/highway#1935).
#
# On Windows, shared_library() produces jvector.dll; the version: keyword is
# silently ignored by Meson on Windows (DLLs have no .so versioning convention),
# so it is safe to pass unconditionally.
vectorutil_lib = shared_library(
  'jvector',
  sources            : ['jvector_simd.cpp',
                        'third_party/highway/hwy/abort.cc'],
  include_directories: [include_directories('.'), hwy_inc],
  cpp_args           : ['-DJVECTOR_BUILD'] + visibility_flag,
  link_whole         : isa_libs,
  version            : meson.project_version(),
  install            : true,
)

# Dependency object for use by executables/tests in this build tree.
vectorutil_dep = declare_dependency(
  link_with          : vectorutil_lib,
  include_directories: include_directories('.'),
)

## Example driver that exercises the runtime-dispatch API.
#executable(
#  'cpp_driver',
#  sources     : 'examples/cpp_driver.cpp',
#  dependencies: vectorutil_dep,
#)
#
## ---- Tests -----------------------------------------------------------------
#gtest_dep = dependency('gtest_main', required: true)
#
#test_exe = executable(
#  'test_kernels',
#  sources     : [
#    'tests/test_kernels.cpp',
#    'tests/test_cpuFeatures.cpp',
#  ],
#  dependencies: [vectorutil_dep, gtest_dep],
#)
#
#test('kernels',     test_exe, protocol: 'gtest', suite: 'kernels')
#test('cpu_features', test_exe, protocol: 'gtest', suite: 'cpu',
#     args: ['--gtest_filter=CpuFeaturesTest.*'])
#
## ---- Benchmarks ------------------------------------------------------------
#gbench_dep = dependency('benchmark', required: false)
#if gbench_dep.found()
#  executable(
#    'bench_kernels',
#    sources     : 'benchmarks/bench_kernels.cpp',
#    dependencies: [vectorutil_dep, gbench_dep],
#    cpp_args    : ['-O3'],
#  )
#endif
#
#rust_enabled = add_languages('rust', required: false)
#if rust_enabled
#  executable(
#    'rust_driver',
#    'examples/rust_driver.rs',
#    link_with: vectorutil_lib,
#  )
#endif
#