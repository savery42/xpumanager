/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_LIN_PRIVILEGE_DETAIL_H
#define OAL_LIN_PRIVILEGE_DETAIL_H

#include "privilege.h"
#include <cstdint>
#include <istream>
#include <optional>
#include <vector>

/// The policy behind currentPrivileges(), split from the I/O so tests can drive it directly.
namespace privilege_detail {

/// What the kernel checks for this process, gathered once.
struct Credentials
{
	std::uint32_t euid{};
	std::vector<std::uint32_t> groups;	  ///< effective gid and supplementary groups of this process
	std::optional<std::uint64_t> capEff;  ///< effective capability mask; nullopt when unreadable
	std::optional<std::uint32_t> xpumGid; ///< gid of the 'xpum' group, if it exists
	std::optional<int> xeObservation;	  ///< dev.xe.observation_paranoid, if present
	std::optional<int> i915PerfStream;	  ///< dev.i915.perf_stream_paranoid, if present
	std::optional<int> perfEventParanoid; ///< kernel.perf_event_paranoid, if readable
};

/**
 * @brief Maps credentials to the capabilities they grant.
 *
 * @param[in] c  Credentials of the process to evaluate.
 * @return The capabilities held.
 */
[[nodiscard]] Capability evaluate(const Credentials &c);

/**
 * @brief Reads the CapEff mask from /proc/<pid>/status content.
 *
 * @param[in] status  Stream positioned at the start of the status text.
 * @retval mask       The hexadecimal CapEff value.
 * @retval nullopt    No CapEff line, or its value is not hexadecimal.
 */
[[nodiscard]] std::optional<std::uint64_t> parseCapEff(std::istream &status);

} // namespace privilege_detail

#endif // OAL_LIN_PRIVILEGE_DETAIL_H
