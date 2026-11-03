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
#include "cmd_ps.h"
#include "cmd_stats.h"
#include "cmd_top.h"
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
	TEST_CASE("device-facing commands default to sysman only")
	{
		const cmdDiscovery discovery;
		CHECK(discovery.plan(Argv{}.view()).driver == DriverMode::SysmanOnly);
		CHECK(discovery.plan(Argv{"--pf", "-j"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdVgpu{}.plan(Argv{"-l", "-d", "0"}.view()).driver == DriverMode::SysmanOnly);
	}

	TEST_CASE("only properties, P2P and EU without a sysman source need the compute runtime")
	{
		const cmdDiscovery discovery;
		CHECK(discovery.plan(Argv{"-d", "0", "-j"}.view()).driver == DriverMode::Full);
		CHECK(discovery.plan(Argv{"--dump", "30"}.view()).driver == DriverMode::Full);
		CHECK(discovery.plan(Argv{"-d", "0", "--listamcversions"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdTopology{}.plan(Argv{"--p2p", "r"}.view()).driver == DriverMode::Full);
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

	TEST_CASE("the same commands need sysman once they have work to do")
	{
		CHECK(cmdDump{}.plan(Argv{"-d", "0", "--metrics", "name"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdHealth{}.plan(Argv{"-d", "0"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdCrashlog{}.plan(Argv{"--extract", "-d", "0"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdTopology{}.plan(Argv{"-m"}.view()).driver == DriverMode::SysmanOnly);
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
		for (const char *form : {"--reset", "--reset=true", "--resetfrequencyrange", "--coldreset"}) {
			CAPTURE(form);
			CHECK(config.plan(Argv{"-d", "0", form}.view()).driver == DriverMode::SysmanOnly);
		}
	}

	TEST_CASE("a spelling CLI11 accepts is seen the same way")
	{
		CHECK(cmdTopology{}.plan(Argv{"--p2p=r"}.view()).driver == DriverMode::Full);
		CHECK(cmdDiscovery{}.plan(Argv{"--device=0"}.view()).driver == DriverMode::Full);
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

	TEST_CASE("dump: every field is served by sysman")
	{
		const cmdDump dump;
		for (const char *fields :
			 {"name", "uuid", "driver_version", "pci.device_id", "IDENTITY", "power.draw", "1", "eu.active"}) {
			CAPTURE(fields);
			CHECK(dump.plan(Argv{"--metrics", fields}.view()).driver == DriverMode::SysmanOnly);
		}
		CHECK(dump.plan(Argv{"--metrics=uuid", "--loop-ms", "50"}.view()).driver == DriverMode::SysmanOnly);
	}

	TEST_CASE("dump: forms run() rejects before touching a device need no driver")
	{
		const cmdDump dump;
		CHECK(dump.plan(Argv{"-d", "0"}.view()).driver == DriverMode::None);			  // help: no --metrics
		CHECK(dump.plan(Argv{"--metrics", "no.such"}.view()).driver == DriverMode::None); // unresolved
		CHECK(dump.plan(Argv{"--metrics"}.view()).driver == DriverMode::None);			  // parse error
		CHECK(dump.plan(Argv{"-d", "0", "help"}.view()).driver == DriverMode::None);
	}

	TEST_CASE("top-level queries use the same field-derived plan")
	{
		CHECK(cmdDump::queryPlan("name,power.draw").driver == DriverMode::SysmanOnly);
		CHECK(cmdDump::queryPlan("no.such").driver == DriverMode::None);
		CHECK(cmdDump::queryPlan("eu.stall").optional.size() == 1);
		CHECK(cmdDump::queryPlan("eu.stall").optional.front().announce);
		// utilization.gpu reads the busiest engine, not the EU array, so it needs no EU feature.
		CHECK(cmdDump::queryPlan("utilization.gpu").driver == DriverMode::SysmanOnly);
		CHECK(cmdDump::queryPlan("utilization.gpu").optional.empty());
		CHECK(cmdDump::queryPlan("utilization.gpu,eu.active").optional.front().announce);
	}

	TEST_CASE("ps and stats are sysman-only at baseline")
	{
		CHECK(cmdPs{}.plan(Argv{}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdTop{}.plan(Argv{}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdStats{}.plan(Argv{"-d", "0"}.view()).driver == DriverMode::SysmanOnly);
		CHECK(cmdStats{}.plan(Argv{"-d", "0", "--bogus"}.view()).driver == DriverMode::None);
	}
}

TEST_SUITE("cmds::plan EU")
{
	TEST_CASE("ps, stats and dump declare EU as an optional PerfObserve feature that needs Full")
	{
		const auto onlyEu = [](const CommandPlan &plan) {
			return plan.required == Capability::NONE && plan.optional.size() == 1 &&
				   plan.optional.front().capability == Capability::PerfObserve &&
				   plan.optional.front().driver == DriverMode::Full;
		};
		CHECK(onlyEu(cmdPs{}.plan(Argv{}.view())));
		CHECK(onlyEu(cmdTop{}.plan(Argv{}.view())));
		CHECK(cmdTop{}.plan(Argv{"--help"}.view()).driver == DriverMode::None);
		for (const char *eu : {"--eu", "-e", "-je", "--eu=true"}) {
			CAPTURE(eu);
			CHECK(onlyEu(cmdStats{}.plan(Argv{"-d", "0", eu}.view())));
		}
		CHECK(cmdStats{}.plan(Argv{"-d", "0", "--eu=false"}.view()).optional.empty());
		CHECK(cmdStats{}.plan(Argv{"-d", "0"}.view()).optional.empty());
		CHECK(onlyEu(cmdDump{}.plan(Argv{"--metrics", "eu.active,power.draw"}.view())));
		CHECK(onlyEu(cmdDump{}.plan(Argv{"--metrics=eu.stall"}.view())));
		CHECK(cmdDump{}.plan(Argv{"--metrics", "power.draw"}.view()).optional.empty());
	}

	TEST_CASE("without PerfObserve the EU feature is skipped and the driver stays sysman-only")
	{
		const auto plan = cmdStats{}.plan(Argv{"--eu"}.view());
		CHECK(resolve(plan, Capability::NONE).driver == DriverMode::SysmanOnly);
		CHECK(resolve(plan, Capability::NONE).skipped.size() == 1);
		CHECK(resolve(plan, Capability::PerfObserve).driver == DriverMode::Full);
		CHECK(resolve(plan, Capability::PerfObserve).skipped.empty());
	}
}
