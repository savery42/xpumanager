/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "privilege_detail.h"

#include <sstream>
#include <string>

using privilege_detail::Credentials;
using privilege_detail::evaluate;
using privilege_detail::parseCapEff;

namespace {

constexpr std::uint32_t USER_UID = 1000;
constexpr std::uint32_t XPUM_GID = 990;

// Bit 38 is CAP_PERFMON, bit 21 CAP_SYS_ADMIN.
constexpr std::uint64_t CAP_PERFMON_MASK = std::uint64_t{1} << 38;
constexpr std::uint64_t CAP_SYS_ADMIN_MASK = std::uint64_t{1} << 21;

/// An ordinary user on a box with every knob at its restrictive default.
[[nodiscard]] Credentials user()
{
	return {.euid = USER_UID,
			.groups = {USER_UID},
			.capEff = 0,
			.xpumGid = XPUM_GID,
			.xeObservation = 1,
			.i915PerfStream = std::nullopt,
			.perfEventParanoid = 2};
}

[[nodiscard]] std::optional<std::uint64_t> capEffOf(const std::string &status)
{
	std::istringstream in{status};
	return parseCapEff(in);
}

} // namespace

TEST_CASE("Capability set arithmetic")
{
	constexpr auto perf = Capability::PerfObserve;
	CHECK(hasAll(perf, Capability::PerfObserve));
	CHECK_FALSE(hasAll(perf, Capability::Admin));

	constexpr auto need = Capability::Admin | Capability::PerfObserve;
	CHECK(hasAny(perf, need));
	CHECK_FALSE(hasAll(perf, need));
	CHECK(missing(perf, need) == Capability::Admin);
	CHECK(missing(Capability::ALL, need) == Capability::NONE);
	CHECK(missing(Capability::NONE, Capability::NONE) == Capability::NONE);
}

TEST_CASE("an ordinary user holds nothing") { CHECK(evaluate(user()) == Capability::NONE); }

TEST_CASE("Admin: effective uid 0 or the xpum group of this process")
{
	auto c = user();
	c.groups.push_back(XPUM_GID);
	CHECK(evaluate(c) == Capability::Admin);

	c = user();
	c.xpumGid = std::nullopt; // no such group on this system
	c.groups.push_back(XPUM_GID);
	CHECK(evaluate(c) == Capability::NONE);

	// A root whose capabilities were dropped still owns root's sysfs files.
	c = user();
	c.euid = 0;
	CHECK(evaluate(c) == Capability::Admin);
}

TEST_CASE("perfmon_capable(): CAP_PERFMON or CAP_SYS_ADMIN grants both perf capabilities")
{
	auto c = user();
	c.capEff = CAP_PERFMON_MASK;
	CHECK(evaluate(c) == (Capability::PerfObserve | Capability::PerfEvents));

	c.capEff = CAP_SYS_ADMIN_MASK;
	CHECK(evaluate(c) == (Capability::PerfObserve | Capability::PerfEvents));
}

TEST_CASE("unreadable CapEff: euid 0 stands in for every capability, others get none")
{
	auto c = user();
	c.capEff = std::nullopt;
	CHECK(evaluate(c) == Capability::NONE);

	c.euid = 0;
	CHECK(evaluate(c) == Capability::ALL);
}

TEST_CASE("observation knobs grant PerfObserve only when every present one is open")
{
	auto c = user();
	c.xeObservation = 0;
	CHECK(evaluate(c) == Capability::PerfObserve);

	c.i915PerfStream = 1;
	CHECK(evaluate(c) == Capability::NONE);

	c.xeObservation = std::nullopt;
	c.i915PerfStream = 0;
	CHECK(evaluate(c) == Capability::PerfObserve);

	c.i915PerfStream = std::nullopt; // no GPU observation driver loaded at all
	CHECK(evaluate(c) == Capability::NONE);
}

TEST_CASE("perf_event_paranoid <= 0 grants PerfEvents without capabilities")
{
	auto c = user();
	c.perfEventParanoid = 1;
	CHECK(evaluate(c) == Capability::NONE);

	c.perfEventParanoid = 0;
	CHECK(evaluate(c) == Capability::PerfEvents);

	c.perfEventParanoid = -1;
	CHECK(evaluate(c) == Capability::PerfEvents);

	c.perfEventParanoid = std::nullopt;
	CHECK(evaluate(c) == Capability::NONE);
}

TEST_CASE("parseCapEff reads the hexadecimal CapEff line")
{
	CHECK(capEffOf("Name:\ttest\nCapEff:\t0000004000000000\nCapBnd:\tffffffffffffffff\n") == CAP_PERFMON_MASK);
	CHECK(capEffOf("CapEff:\t0000000000000000\n") == std::uint64_t{0});
	CHECK_FALSE(capEffOf("CapEff:\tnot-hex\n").has_value());
	CHECK_FALSE(capEffOf("CapEff:\t00ff trailing\n").has_value());
	CHECK_FALSE(capEffOf("CapEff:\n").has_value());
	CHECK_FALSE(capEffOf("Name:\ttest\n").has_value());
}
