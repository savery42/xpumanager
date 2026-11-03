# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT
#
# Integration test helpers: standalone Level Zero programs used by the YAML
# validation suite (validation/tests/) to create controlled GPU workloads.
#

process_memstress_exe = executable(
  'process_memstress',
  'process_memstress.cpp',
  include_directories: [global_inc],
  dependencies: [levelzero_dep, compat_dep],
  link_args: ['-pie'],
  build_by_default: true,
  install: false,
)

cross_gpu_alloc_exe = executable(
  'cross_gpu_alloc',
  'cross_gpu_alloc.cpp',
  include_directories: [global_inc],
  dependencies: [levelzero_dep, compat_dep],
  link_args: ['-pie'],
  build_by_default: true,
  install: false,
)

engine_context_probe_exe = executable(
  'engine_context_probe',
  'engine_context_probe.cpp',
  include_directories: [global_inc],
  dependencies: [levelzero_dep, compat_dep],
  link_args: ['-pie'],
  build_by_default: true,
  install: false,
)

# OpenCL load generator (compute, memory, copy, bus) for any GPU with an OpenCL driver; see
# gpu_stress --help. Built only where OpenCL and std::format are available.
opencl_dep = dependency('OpenCL', required: false)
if opencl_dep.found() and cpp.has_header('format')
  gpu_stress_exe = executable(
    'gpu_stress',
    'gpu_stress.cpp',
    cpp_args: ['-DCL_TARGET_OPENCL_VERSION=300'],
    dependencies: [opencl_dep, thread_dep],
    link_args: ['-pie'],
    build_by_default: true,
    install: false,
  )
endif
