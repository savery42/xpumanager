/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "requirements.h"

#include <array>

namespace {

constexpr auto ADMIN = Capability::Admin;
constexpr auto PERF = Capability::PerfObserve;

constexpr std::array FEATURES{
	OptionalFeature{"EU columns", PERF, DriverMode::Full},
	OptionalFeature{"cold reset", ADMIN | PERF, DriverMode::None},
};

constexpr CommandPlan PLAN{.driver = DriverMode::SysmanOnly, .optional = FEATURES};

} // namespace

TEST_CASE("a missing required capability refuses the plan")
{
	const CommandPlan plan{.required = ADMIN};
	CHECK_FALSE(resolve(plan, PERF).permitted());
	CHECK(resolve(plan, PERF).missing == ADMIN);
	CHECK(resolve(plan, ADMIN | PERF).permitted());
	CHECK(resolve(CommandPlan{}, Capability::NONE).permitted());
}

TEST_CASE("an available optional feature raises the driver; a missing one is skipped")
{
	const auto none = resolve(PLAN, Capability::NONE);
	CHECK(none.driver == DriverMode::SysmanOnly);
	REQUIRE(none.skipped.size() == 2);
	CHECK(none.skipped[0].name == "EU columns");
	CHECK(none.skipped[1].name == "cold reset");

	const auto perf = resolve(PLAN, PERF);
	CHECK(perf.driver == DriverMode::Full);
	REQUIRE(perf.skipped.size() == 1);
	CHECK(perf.skipped[0].name == "cold reset");
}

TEST_CASE("a feature that needs two capabilities needs both")
{
	CHECK(resolve(PLAN, ADMIN).skipped.size() == 2);
	CHECK(resolve(PLAN, ADMIN | PERF).skipped.empty());
}

TEST_CASE("the driver is never lowered below the baseline")
{
	const CommandPlan plan{.driver = DriverMode::Full, .optional = FEATURES};
	CHECK(resolve(plan, Capability::ALL).driver == DriverMode::Full);
}

TEST_CASE("a feature with a fallback is still skipped, but marked silent")
{
	constexpr std::array silent{
		OptionalFeature{.name = "x", .capability = PERF, .driver = DriverMode::Full, .announce = false}};
	const auto r = resolve(CommandPlan{.optional = silent}, Capability::NONE);
	REQUIRE(r.skipped.size() == 1);
	CHECK_FALSE(r.skipped.front().announce);
	CHECK(resolve(CommandPlan{.optional = silent}, PERF).driver == DriverMode::Full);
}

TEST_CASE("describeAll names each capability once, lowest bit first")
{
	CHECK(describeAll(Capability::NONE).empty());
	CHECK(describeAll(ADMIN) == describe(ADMIN));
	CHECK(describeAll(ADMIN | PERF) == std::string{describe(ADMIN)} + "; and " + std::string{describe(PERF)});
}
