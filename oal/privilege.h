/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#ifndef OAL_PRIVILEGE_H
#define OAL_PRIVILEGE_H

#include "utility/bitmask_enum.h"
#include <cstdint>
#include <string>
#include <string_view>

/**
 * @brief A distinct kind of privilege a command may depend on; also a set of them.
 *
 * Distinct from KernelCapability below, which names a single kernel capability bit for
 * explaining why one value could not be read.
 *
 * "Privileged" is not one thing: each capability is granted by a different
 * mechanism, so a user can hold one and not another. Combine with `|`.
 */
enum class Capability : std::uint8_t
{
	NONE = 0,
	Admin = 1U << 0,	   ///< root or the 'xpum' group: sysman writes, VF create/remove, cold reset
	PerfObserve = 1U << 1, ///< GPU observation (OA) streams: EU metrics
	PerfEvents = 1U << 2,  ///< system-wide perf events: the GPU PMU behind VF engine utilisation
	ALL = Admin | PerfObserve | PerfEvents,
};

consteval bool enableBitmaskOperators(Capability) { return true; }

/// The capabilities in @p need that @p held does not include.
[[nodiscard]] constexpr Capability missing(Capability held, Capability need) noexcept { return need & ~held; }

/**
 * @brief How a user can obtain a single capability, phrased for a diagnostic message.
 *
 * @retval ""  @p c is NONE or a set of more than one capability.
 */
[[nodiscard]] constexpr std::string_view describe(Capability c) noexcept
{
	switch (c) {
	case Capability::Admin:
		return "root or membership of the 'xpum' group";
	case Capability::PerfObserve:
		return "root, CAP_PERFMON, or sysctl dev.xe.observation_paranoid=0 (dev.i915.perf_stream_paranoid=0 on i915)";
	case Capability::PerfEvents:
		return "root, CAP_PERFMON, or sysctl kernel.perf_event_paranoid=0";
	default:
		return {};
	}
}

/**
 * @brief Reports which capabilities the calling process holds.
 *
 * Makes no driver calls, so it is safe to call before any Level Zero
 * initialisation. A source that cannot be read counts as "not held". This is
 * a pre-check: the kernel can still deny an operation the result says is
 * permitted (LSMs, device node permissions).
 *
 * On Windows every capability is reported as held.
 *
 * @return The capabilities held.
 */
[[nodiscard]] Capability currentPrivileges();

/** A Linux capability a read can need, for saying why a value is missing. */
enum class KernelCapability : std::uint8_t
{
	SysAdmin, ///< CAP_SYS_ADMIN: PCI configuration space past its first 64 bytes, among others
	Perfmon,  ///< CAP_PERFMON: the engine busyness counters
};

/**
 * @brief The capability's name as the kernel spells it, e.g. "CAP_SYS_ADMIN".
 *
 * @param[in] cap The capability.
 *
 * @retval name The name.
 */
[[nodiscard]] constexpr std::string_view capabilityName(KernelCapability cap) noexcept
{
	return cap == KernelCapability::SysAdmin ? "CAP_SYS_ADMIN" : "CAP_PERFMON";
}

/**
 * @brief Whether the process holds @p cap in its effective set.
 *
 * As the kernel does, CAP_SYS_ADMIN also grants what CAP_PERFMON does.
 *
 * @param[in] cap The capability.
 *
 * @retval true  The process holds it, or the OS has no such capability (Windows).
 * @retval false The process lacks it.
 */
[[nodiscard]] bool hasCapability(KernelCapability cap) noexcept;

/**
 * @brief Whether the process may open @p path for reading and writing, as a device node needs.
 *
 * @param[in] path The file.
 *
 * @retval true  It may, or the check does not apply (Windows).
 * @retval false It may not, or the file does not exist.
 */
[[nodiscard]] bool canReadWrite(const std::string &path) noexcept;

#endif // OAL_PRIVILEGE_H
