/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "privilege.h"
#include "privilege_detail.h"

#include <algorithm>
#include <charconv>
#include <fcntl.h>
#include <fstream>
#include <grp.h>
#include <optional>
#include <string>
#include <string_view>
#include <unistd.h>

namespace {

// Bit positions in the CapEff mask (linux/capability.h).
constexpr unsigned CAP_SYS_ADMIN_BIT = 21;
constexpr unsigned CAP_PERFMON_BIT = 38;

[[nodiscard]] constexpr bool hasBit(std::uint64_t mask, unsigned bit) noexcept { return ((mask >> bit) & 1U) != 0; }

} // namespace

namespace privilege_detail {

Capability evaluate(const Credentials &c)
{
	Capability held = Capability::NONE;

	// Sysman writes go to root-owned sysfs files, which the kernel opens for the
	// owner without any capability check, so euid 0 is enough even when the
	// effective capability set has been dropped (a container's root).
	const bool inXpumGroup = c.xpumGid && std::ranges::find(c.groups, *c.xpumGid) != c.groups.end();
	if (c.euid == 0 || inXpumGroup) {
		held |= Capability::Admin;
	}

	// Observation and perf events are capability checks: perfmon_capable() is
	// CAP_PERFMON or CAP_SYS_ADMIN, so a root without them is not trusted. Only
	// when CapEff cannot be read does euid 0 stand in for the full set.
	const std::uint64_t caps = c.capEff.value_or(c.euid == 0 ? ~std::uint64_t{0} : 0);
	const bool perfmonCapable = hasBit(caps, CAP_PERFMON_BIT) || hasBit(caps, CAP_SYS_ADMIN_BIT);

	// The knob of each loaded driver must allow it: a device is driven by only
	// one of them, and which one is not known before driver init.
	const bool anyObservationKnob = c.xeObservation || c.i915PerfStream;
	const bool observationOpen =
		anyObservationKnob && c.xeObservation.value_or(0) == 0 && c.i915PerfStream.value_or(0) == 0;
	if (perfmonCapable || observationOpen) {
		held |= Capability::PerfObserve;
	}

	// A system-wide event needs perf_event_paranoid <= 0 without perfmon_capable().
	if (perfmonCapable || (c.perfEventParanoid && *c.perfEventParanoid <= 0)) {
		held |= Capability::PerfEvents;
	}
	return held;
}

std::optional<std::uint64_t> parseCapEff(std::istream &status)
{
	constexpr std::string_view key = "CapEff:";
	for (std::string line; std::getline(status, line);) {
		if (!line.starts_with(key)) {
			continue;
		}
		const std::size_t start = line.find_first_not_of(" \t", key.size());
		if (start == std::string::npos) {
			return std::nullopt;
		}
		std::uint64_t mask = 0;
		const char *last = line.data() + line.size();
		const auto [end, ec] = std::from_chars(line.data() + start, last, mask, 16);
		if (ec != std::errc{} || end != last) {
			return std::nullopt;
		}
		return mask;
	}
	return std::nullopt;
}

} // namespace privilege_detail

namespace {

constexpr const char XPUM_GROUP[] = "xpum"; // NOLINT(modernize-avoid-c-arrays): getgrnam needs a C string

/// A small signed integer sysctl, or nullopt when absent or unreadable.
[[nodiscard]] std::optional<int> readSysctl(const char *path)
{
	std::ifstream in{path};
	int value = 0;
	if (in >> value) {
		return value;
	}
	return std::nullopt;
}

/// The effective gid and every supplementary group of this process, as the kernel sees them.
[[nodiscard]] std::vector<std::uint32_t> processGroups()
{
	const int count = getgroups(0, nullptr);
	std::vector<gid_t> groups(static_cast<std::size_t>(std::max(count, 0)));
	const int filled = groups.empty() ? 0 : getgroups(count, groups.data());
	groups.resize(static_cast<std::size_t>(std::max(filled, 0)));
	groups.push_back(getegid());
	return {groups.begin(), groups.end()};
}

[[nodiscard]] std::optional<std::uint32_t> xpumGid()
{
	const group *xpum = getgrnam(XPUM_GROUP);
	return xpum != nullptr ? std::optional<std::uint32_t>{xpum->gr_gid} : std::nullopt;
}

[[nodiscard]] std::optional<std::uint64_t> readCapEff()
{
	std::ifstream status{"/proc/self/status"};
	return privilege_detail::parseCapEff(status);
}

} // namespace

Capability currentPrivileges()
{
	return privilege_detail::evaluate({
		.euid = geteuid(),
		.groups = processGroups(),
		.capEff = readCapEff(),
		.xpumGid = xpumGid(),
		.xeObservation = readSysctl("/proc/sys/dev/xe/observation_paranoid"),
		.i915PerfStream = readSysctl("/proc/sys/dev/i915/perf_stream_paranoid"),
		.perfEventParanoid = readSysctl("/proc/sys/kernel/perf_event_paranoid"),
	});
}

bool hasCapability(KernelCapability cap) noexcept
{
	try {
		// Read once: the effective set does not change while xpu-smi runs.
		static const std::optional<std::uint64_t> caps = readCapEff();
		if (!caps) {
			return geteuid() == 0;
		}
		const bool sysAdmin = hasBit(*caps, CAP_SYS_ADMIN_BIT);
		return cap == KernelCapability::SysAdmin ? sysAdmin : sysAdmin || hasBit(*caps, CAP_PERFMON_BIT);
	} catch (...) {
		return geteuid() == 0;
	}
}

bool canReadWrite(const std::string &path) noexcept
{
	return faccessat(AT_FDCWD, path.c_str(), R_OK | W_OK, AT_EACCESS) == 0;
}
