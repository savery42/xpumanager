# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: MIT

# Unit tests for XPU Manager components

if get_option('with_tests')
  doctest_dep = dependency('doctest', required: true)

  # OAL diagnostics log tests
  # Tests filesystem operations, stream handling, and error scenarios
  dbg_log_test = executable(
    'dbg_log_test',
    'dbg_log_test.cpp',
    include_directories: [global_inc],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  # Register tests with meson
  test('dbg_log_tests', dbg_log_test)

  # Crash log pure helper tests (command building and iclg output parsing).
  # Compiles crashlog_lin.cpp directly so the test needs no OAL link.
  crashlog_lin_test = executable(
    'crashlog_lin_test',
    ['crashlog_lin_test.cpp', '../crashlog_lin.cpp'],
    include_directories: [global_inc, include_directories('..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('crashlog_lin_test', crashlog_lin_test)

  # $PATH executable lookup tests. Compiles path_lookup.cpp directly so the
  # test needs no OAL link.
  path_lookup_test = executable(
    'path_lookup_test',
    ['path_lookup_test.cpp', '../path_lookup.cpp'],
    include_directories: [global_inc, include_directories('..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('path_lookup_test', path_lookup_test)

  # Terminal size tests, against a pseudo-terminal. Compiles terminal.cpp directly so the
  # test needs no OAL link.
  terminal_test = executable(
    'terminal_test',
    ['terminal_test.cpp', '../terminal.cpp'],
    include_directories: [global_inc, include_directories('..'), include_directories('../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('terminal_test', terminal_test)
  pci_switch_test = executable(
    'pci_switch_test',
    'pci_switch_test.cpp',
    include_directories: [include_directories('..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('pci_switch_tests', pci_switch_test)

  # Kernel-mode driver identification tests (driver resolution and srcversion
  # reads against a fake sysfs tree). Compiles kernel_driver.cpp directly so the
  # test needs no OAL link.
  kernel_driver_test = executable(
    'kernel_driver_test',
    ['kernel_driver_test.cpp', '../kernel_driver.cpp'],
    include_directories: [global_inc, include_directories('..')],
    dependencies: [doctest_dep, bdf_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('kernel_driver_test', kernel_driver_test)

  # PCI subsystem device ID and class code reads against a fake sysfs tree. Compiles
  # pci_sysfs.cpp directly so the test needs no OAL link.
  pci_sysfs_test = executable(
    'pci_sysfs_test',
    ['pci_sysfs_test.cpp', '../pci_sysfs.cpp'],
    include_directories: [global_inc, include_directories('..'), include_directories('../..')],
    dependencies: [doctest_dep, bdf_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('pci_sysfs_test', pci_sysfs_test)

  # VRAM manager debugfs path resolution tests. Compiles vram_debugfs.cpp
  # directly so the test needs no OAL link.
  vram_debugfs_test = executable(
    'vram_debugfs_test',
    ['vram_debugfs_test.cpp', '../vram_debugfs.cpp'],
    include_directories: [global_inc, include_directories('..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('vram_debugfs_test', vram_debugfs_test)

  # parsePchResponse pure-logic tests (header validation, byte extraction, state
  # mapping). Header-only function — no METEE linkage required.
  pch_prod_state_test = executable(
    'pch_prod_state_test',
    'pch_prod_state_test.cpp',
    include_directories: [global_inc, include_directories('..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('pch_prod_state_test', pch_prod_state_test)

  # Process platform fdinfo tests (vramFromFdinfo, fixProcessMemSize).
  # Compiles process_platform.cpp directly so the test needs no OAL link;
  # proc_fdinfo.cpp comes with it for fdinfo::engineFlagForKey.
  process_platform_test = executable(
    'process_platform_test',
    ['process_platform_test.cpp', '../process_platform.cpp', '../proc_fdinfo.cpp'],
    include_directories: [global_inc, include_directories('..'), include_directories('../..'), include_directories('../../../hal/core')],
    dependencies: [doctest_dep, levelzero_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('process_platform_test', process_platform_test)

  # Per-device firmware update lock. Needs root; reports itself skipped otherwise.
  # Compiles fs_lock.cpp directly so the test needs no OAL link.
  fs_lock_test = executable(
    'fs_lock_test',
    ['fs_lock_test.cpp', '../fs_lock.cpp'],
    include_directories: [global_inc, include_directories('..'), include_directories('../..')],
    dependencies: [doctest_dep, bdf_dep, compat_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('fs_lock_test', fs_lock_test)

  # Driver library version parsing; compiles dynload.cpp directly.
  dynload_test = executable(
    'dynload_test',
    ['dynload_test.cpp', '../dynload.cpp'],
    include_directories: [global_inc, include_directories('../..')],
    dependencies: [doctest_dep, cpp.find_library('dl', required: false)],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('dynload_test', dynload_test)

  # Privilege policy tests: credentials in, capabilities out, plus CapEff parsing.
  # Compiles privilege.cpp directly so the test needs no OAL link.
  privilege_test = executable(
    'privilege_test',
    ['privilege_test.cpp', '../privilege.cpp'],
    include_directories: [global_inc, include_directories('..', '../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('privilege_test', privilege_test)

  # Key decoding tests. decodeKey() is constexpr and header-only, so the test needs no OAL link.
  keyboard_test = executable(
    'keyboard_test',
    'keyboard_test.cpp',
    include_directories: [global_inc, include_directories('../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('keyboard_test', keyboard_test)

  # Process signalling tests against a forked child. Compiles process_control.cpp directly so the test needs no OAL link.
  process_control_test = executable(
    'process_control_test',
    ['process_control_test.cpp', '../process_control.cpp'],
    include_directories: [global_inc, include_directories('../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('process_control_test', process_control_test)

  # Local time formatting tests with TZ set to POSIX rule strings. Compiles local_time.cpp directly so the test needs no OAL link.
  local_time_test = executable(
    'local_time_test',
    ['local_time_test.cpp', '../local_time.cpp'],
    include_directories: [global_inc, include_directories('../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('local_time_test', local_time_test)

  # Process CPU time and resident memory from fixture stat files. Compiles process_stats.cpp directly so the test needs no OAL link.
  process_stats_test = executable(
    'process_stats_test',
    ['process_stats_test.cpp', '../process_stats.cpp'],
    include_directories: [global_inc, include_directories('../..')],
    dependencies: [doctest_dep],
    link_args: is_linux ? ['-pie'] : [],
    build_by_default: true,
  )

  test('process_stats_test', process_stats_test)

  message('Unit tests enabled for OAL diagnostics')
else
  message('Skipping OAL diagnostic tests (pass -Dwith_tests=true to enable)')
endif