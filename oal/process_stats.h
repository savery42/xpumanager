/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_PROCESS_STATS_H
#define OAL_PROCESS_STATS_H

#include "utility/units/units.h"

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

/** What the host side of a process is using, read once. */
struct ProcessHostSample
{
	std::chrono::nanoseconds cpuTime; ///< user + system time over the life of the process, all threads
	xpum::units::Bytes resident;	  ///< resident set size
};

/**
 * @brief Reads a process's accumulated CPU time and resident memory.
 *
 * On Linux this is /proc/<pid>/stat, which any user may read for any process (unless /proc
 * is mounted with hidepid). CPU use is the change in @c cpuTime between two reads.
 *
 * @param[in] pid       Process to read.
 * @param[in] procRoot  Defaults to "/proc"; override in tests to point at a fixture directory.
 *
 * @retval sample   The process exists and its stat line parsed.
 * @retval nullopt  It has exited, is hidden, the line is malformed, or the platform has no support.
 */
[[nodiscard]] std::optional<ProcessHostSample> readProcessHostSample(std::uint32_t pid,
																	 const std::string &procRoot = "/proc");

#endif // OAL_PROCESS_STATS_H
