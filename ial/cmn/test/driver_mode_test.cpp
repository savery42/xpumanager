/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#ifdef INFO
#undef INFO
#endif

#include <string>
std::string progName = "test";

#include "cmd_amc.h"
#include "cmd_config.h"
#include "cmd_crashlog.h"
#include "cmd_discovery.h"
#include "cmd_dump.h"
#include "cmd_health.h"
#include "cmd_listpciinfo.h"
#include "cmd_log.h"
#include "cmd_topology.h"
#include "cmd_vgpu.h"

#ifdef INFO
#undef INFO
#endif

#include <initializer_list>
#include <span>
#include <vector>

namespace {

// Owns the mutable char * tokens that commands receive as argv.
class Argv
{
	std::vector<std::string> tokens;
	std::vector<char *> ptrs;

public:
	Argv(std::initializer_list<const char *> list) : tokens(list.begin(), list.end())
	{
		for (auto &token : tokens) {
			ptrs.push_back(token.data());
		}
	}
	[[nodiscard]] std::span<char *const> view() const { return ptrs; }
};

} // namespace

// plan() is what the dispatcher consults before run(), so these cases cover
// the same selection runCli() makes.
TEST_SUITE("cmds::plan driver")
{
	TEST_CASE("device-facing commands default to the full driver")
	{
		const cmdDiscovery discovery;
		CHECK(discovery.plan(Argv{}.view()).driver == DriverMode::Full);
		CHECK(discovery.plan(Argv{"-d", "0", "-j"}.view()).driver == DriverMode::Full);
	}

	TEST_CASE("-h and --help never need the driver")
	{
		const cmdDiscovery discovery;
		CHECK(discovery.plan(Argv{"-h"}.view()).driver == DriverMode::None);
		CHECK(discovery.plan(Argv{"-d", "0", "--help"}.view()).driver == DriverMode::None);
		CHECK(cmdConfig{}.plan(Argv{"--reset", "-h"}.view()).driver == DriverMode::None);
	}

	TEST_CASE("command-owned help forms need no driver")
	{
		CHECK(cmdDump{}.plan(Argv{}.view()).driver == DriverMode::None);
		CHECK(cmdDump{}.plan(Argv{"help"}.view()).driver == DriverMode::None);
		CHECK(cmdHealth{}.plan(Argv{}.view()).driver == DriverMode::None);
		CHECK(cmdCrashlog{}.plan(Argv{}.view()).driver == DriverMode::None);
		CHECK(cmdTopology{}.plan(Argv{}.view()).driver == DriverMode::None);
	}

	TEST_CASE("the same commands need the driver once they have work to do")
	{
		CHECK(cmdDump{}.plan(Argv{"-d", "0", "--metrics", "0"}.view()).driver == DriverMode::Full);
		CHECK(cmdDump{}.plan(Argv{"-d", "0", "help"}.view()).driver == DriverMode::Full);
		CHECK(cmdHealth{}.plan(Argv{"-d", "0"}.view()).driver == DriverMode::Full);
		CHECK(cmdCrashlog{}.plan(Argv{"--extract", "-d", "0"}.view()).driver == DriverMode::Full);
		CHECK(cmdTopology{}.plan(Argv{"-m"}.view()).driver == DriverMode::Full);
	}

	TEST_CASE("commands that never touch Level Zero need no driver")
	{
		const Argv args{"-d", "0", "-j"};
		CHECK(cmdAmc{}.plan(args.view()).driver == DriverMode::None);
		CHECK(cmdListpciinfo{}.plan(args.view()).driver == DriverMode::None);
		CHECK(cmdLogs{}.plan(Argv{"-f", "logs.tar.gz"}.view()).driver == DriverMode::None);
	}

	TEST_CASE("config keeps the compute runtime off a device it is about to reset")
	{
		const cmdConfig config;
		CHECK(config.plan(Argv{"-d", "0", "--reset"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(config.plan(Argv{"--reset", "-d", "0"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(config.plan(Argv{"-d", "0"}.view()).driver == DriverMode::Full);
		CHECK(config.plan(Argv{"-d", "0", "--resetfrequencyrange"}.view()).driver == DriverMode::Full);
		CHECK(config.plan(Argv{"-d", "0000:03:00.0", "--coldreset"}.view()).driver == DriverMode::Full);
	}

	TEST_CASE("a spelling CLI11 accepts is seen the same way")
	{
		const cmdConfig config;
		CHECK(config.plan(Argv{"-d", "0", "--reset=true"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(config.plan(Argv{"-d", "0", "--reset=false"}.view()).driver == DriverMode::Full);
	}

	TEST_CASE("arguments that do not parse need no driver: run() reports the error first")
	{
		CHECK(cmdConfig{}.plan(Argv{"-d", "0", "--no-such-flag"}.view()).driver == DriverMode::None);
		CHECK(cmdVgpu{}.plan(Argv{"-d"}.view()).driver == DriverMode::None);
	}
}

TEST_SUITE("cmds::plan privilege")
{
	TEST_CASE("vgpu: only create and remove require Admin")
	{
		const cmdVgpu vgpu;
		for (const char *flag : {"-c", "--create", "-r", "--remove", "--create=true"}) {
			CAPTURE(flag);
			CHECK(vgpu.plan(Argv{flag, "-d", "0"}.view()).required == Capability::Admin);
		}
		for (const char *flag : {"-l", "--list", "--precheck", "-s", "--create=false"}) {
			CAPTURE(flag);
			CHECK(vgpu.plan(Argv{flag, "-d", "0"}.view()).required == Capability::NONE);
		}
	}

	TEST_CASE("vgpu: short-flag clusters are read as CLI11 reads them")
	{
		const cmdVgpu vgpu;
		CHECK(vgpu.plan(Argv{"-jc", "-d", "0"}.view()).required == Capability::Admin);
		// -dc is -d with the value "c", not --create.
		CHECK(vgpu.plan(Argv{"-dc"}.view()).required == Capability::NONE);
	}

	TEST_CASE("vgpu: stats skips VF engine utilization without PerfEvents")
	{
		const auto plan = cmdVgpu{}.plan(Argv{"-s", "-d", "0"}.view());
		REQUIRE(plan.optional.size() == 1);
		CHECK(plan.optional.front().capability == Capability::PerfEvents);
		CHECK(plan.optional.front().driver == DriverMode::None);
		CHECK(cmdVgpu{}.plan(Argv{"-l", "-d", "0"}.view()).optional.empty());
	}

	TEST_CASE("config: only --coldreset requires Admin")
	{
		const cmdConfig config;
		CHECK(config.plan(Argv{"-d", "0000:03:00.0", "--coldreset"}.view()).required == Capability::Admin);
		CHECK(config.plan(Argv{"-d", "0", "--reset"}.view()).required == Capability::NONE);
		CHECK(config.plan(Argv{"-d", "0"}.view()).required == Capability::NONE);
	}

	TEST_CASE("help never requires privilege")
	{
		for (const char *help : {"-h", "--help"}) {
			CAPTURE(help);
			CHECK(cmdVgpu{}.plan(Argv{"--create", help}.view()).required == Capability::NONE);
			CHECK(cmdConfig{}.plan(Argv{"--coldreset", help}.view()).required == Capability::NONE);
		}
	}

	TEST_CASE("only resets run on when the driver fails to initialize")
	{
		const cmdConfig config;
		CHECK(config.recoversWithoutDriver(Argv{"-d", "0000:03:00.0", "--reset"}.view()));
		CHECK(config.recoversWithoutDriver(Argv{"-d", "0000:03:00.0", "--coldreset"}.view()));
		CHECK_FALSE(config.recoversWithoutDriver(Argv{"-d", "0"}.view()));
		CHECK_FALSE(config.recoversWithoutDriver(Argv{"-d", "0", "--resetfrequencyrange"}.view()));
		CHECK_FALSE(cmdDiscovery{}.recoversWithoutDriver(Argv{"-d", "0000:03:00.0"}.view()));
	}
}
