/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

/**
 * @file top_test.cpp
 * @brief Unit tests for the pure logic behind `xpu-smi top` (top_model.h).
 *
 * Nothing here touches a GPU or the terminal. The TEST_CASE names state each covered behaviour.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#ifdef INFO
#undef INFO
#endif

#include <string>
std::string progName = "test";

#include "process_runtime.h"
#include "top_model.h"
#include "utility/bitmask_enum.h"
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

using namespace top;

namespace {

psInfo makeProc(uint32_t pid, uint64_t memKiB, std::optional<float> compute = std::nullopt)
{
	psInfo p{};
	p.processId = pid;
	p.memSize = memKiB;
	p.util.compute = compute;
	return p;
}

} // namespace

TEST_CASE("SparklineBuffer keeps only the newest samples")
{
	SparklineBuffer<double, 4> buf;
	CHECK(buf.empty());
	for (const double v : {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}) {
		buf.push(v);
	}
	CHECK(buf.count() == 4);
	CHECK(buf.last(4) == std::vector<double>{3.0, 4.0, 5.0, 6.0});
	CHECK(buf.last(2) == std::vector<double>{5.0, 6.0});
	CHECK(buf.last(9).size() == 4);
}

TEST_CASE("sparkLevel never draws a sample as blank, and stops short of the full block")
{
	CHECK(sparkLevel(0.0) == 1);
	CHECK(sparkLevel(50.0) == 4);
	CHECK(sparkLevel(100.0) == 7);
	CHECK(sparkLevel(250.0) == 7);
	CHECK(sparkLevel(-5.0) == 1);
	CHECK(sparkLevel(5.0, 10.0) == 4);
	CHECK(SPARK_BLOCKS[sparkLevel(100.0)] == "\xe2\x96\x87");
}

TEST_CASE("barFill splits a bar into whole cells and an eighth")
{
	CHECK(barFill(0.0, 10) == BarFill{0, 0});
	CHECK(barFill(100.0, 10) == BarFill{10, 0});
	CHECK(barFill(55.0, 10) == BarFill{5, 4});
	CHECK(barFill(3.0, 10) == BarFill{0, 2});
	CHECK(barFill(250.0, 10) == BarFill{10, 0});
	CHECK(barFill(-4.0, 10) == BarFill{0, 0});
	CHECK(barFill(50.0, 0) == BarFill{0, 0});
}

TEST_CASE("parseNumber accepts only a whole number")
{
	CHECK(parseNumber("12.5") == 12.5);
	CHECK(parseNumber("-1") == -1.0);
	CHECK_FALSE(parseNumber("").has_value());
	CHECK_FALSE(parseNumber("N/A").has_value());
	CHECK_FALSE(parseNumber("12x").has_value());
}

TEST_CASE("formatField renders each unit")
{
	CHECK(formatField("32.00", "C") == "32 C");
	CHECK(formatField("62.6", "W") == "62.6 W");
	CHECK(formatField("1200", "MHz") == "1200 MHz");
	CHECK(formatField("12.34", "%") == "12.3%");
	CHECK(formatField("467295.96", "kB/s") == "467.3 MB/s");
	CHECK(formatField("0.1", "MB/s") == "0.1 MB/s");
	CHECK(formatField("32", "MiB") == "32 MiB");
	CHECK(formatField("5", "") == "5");
	CHECK(formatField("2.5", "GT/s") == "2.5 GT/s");
}

TEST_CASE("formatField passes through what is not a positive number")
{
	CHECK(formatField("-1", "") == "N/A");
	CHECK(formatField("None", "") == "None");
	CHECK(formatField("", "%").empty());
}

TEST_CASE("parseSortKey knows exactly four names")
{
	CHECK(parseSortKey("mem") == SortBy::Memory);
	CHECK(parseSortKey("compute") == SortBy::Compute);
	CHECK(parseSortKey("cpu") == SortBy::Cpu);
	CHECK(parseSortKey("pid") == SortBy::Pid);
	CHECK_FALSE(parseSortKey("memory").has_value());
	CHECK_FALSE(parseSortKey("").has_value());
}

TEST_CASE("sortMergedList orders by each key")
{
	const std::vector<std::vector<psInfo>> lists{
		{makeProc(30, 100, 5.0F), makeProc(10, 300)},
		{makeProc(20, 200, 50.0F)},
	};
	const HostUsageMap host{{10, {.cpu = 250.0F, .resident = std::nullopt}},
							{30, {.cpu = 1.0F, .resident = std::nullopt}}};
	auto order = [&](SortBy by) {
		auto merged = buildMergedList(lists, host);
		sortMergedList(merged, by);
		std::vector<uint32_t> pids;
		for (const auto &m : merged) {
			pids.push_back(m.proc->processId);
		}
		return pids;
	};

	CHECK(order(SortBy::Memory) == std::vector<uint32_t>{10, 20, 30});
	CHECK(order(SortBy::Pid) == std::vector<uint32_t>{10, 20, 30});
	// A process with no compute figure sorts last.
	CHECK(order(SortBy::Compute) == std::vector<uint32_t>{20, 30, 10});
	// CPU% can pass 100; a process with no host reading sorts last.
	CHECK(order(SortBy::Cpu) == std::vector<uint32_t>{10, 30, 20});

	const auto merged = buildMergedList(lists, host);
	CHECK(merged[2].gpuIdx == 1);
	CHECK(merged[0].host.cpu == 1.0F);
	CHECK_FALSE(merged[2].host.cpu.has_value());
	CHECK_FALSE(buildMergedList(lists)[0].host.cpu.has_value());
}

TEST_CASE("cpuPercent counts one busy core as 100% and rejects readings it cannot compare")
{
	using namespace std::chrono_literals;
	const auto t0 = std::chrono::steady_clock::time_point{} + 10s;
	const HostReading before{.cpuTime = 5s, .at = t0};

	CHECK(*cpuPercent(before, {.cpuTime = 5s + 500ms, .at = t0 + 1s}) == doctest::Approx(50.0F));
	CHECK(*cpuPercent(before, {.cpuTime = 9s, .at = t0 + 2s}) == doctest::Approx(200.0F));
	CHECK(*cpuPercent(before, {.cpuTime = 5s, .at = t0 + 1s}) == doctest::Approx(0.0F));
	// No time passed, or the pid was reused by a process with less CPU time.
	CHECK_FALSE(cpuPercent(before, {.cpuTime = 6s, .at = t0}).has_value());
	CHECK_FALSE(cpuPercent(before, {.cpuTime = 1s, .at = t0 + 1s}).has_value());
}

TEST_CASE("procTableLayout drops render, copy, GMem%, ShrMiB, SysMiB, then Runtime before the name gets too narrow")
{
	const int allOptional = 3 * OPTIONAL_COL_WIDTH + SHR_COL_WIDTH + SYS_COL_WIDTH + RUNTIME_COL_WIDTH;
	const ProcTableLayout wide = procTableLayout(160);
	CHECK(wide == ProcTableLayout{.showRender = true,
								  .showCopy = true,
								  .showGpuMemPercent = true,
								  .showShared = true,
								  .showSystem = true,
								  .showRuntime = true,
								  .nameWidth = 160 - PROC_FIXED_WIDTH - allOptional});

	// Each threshold is the narrowest width that still keeps the name at MIN_NAME_WIDTH.
	const int onlyRuntime = PROC_FIXED_WIDTH + RUNTIME_COL_WIDTH + MIN_NAME_WIDTH;
	CHECK(procTableLayout(onlyRuntime) == ProcTableLayout{.showRender = false,
														  .showCopy = false,
														  .showGpuMemPercent = false,
														  .showShared = false,
														  .showSystem = false,
														  .showRuntime = true,
														  .nameWidth = MIN_NAME_WIDTH});
	CHECK_FALSE(procTableLayout(onlyRuntime - 1).showRuntime);
	const int withSystem = onlyRuntime + SYS_COL_WIDTH;
	CHECK(procTableLayout(withSystem).showSystem);
	CHECK_FALSE(procTableLayout(withSystem - 1).showSystem);
	const int withShared = withSystem + SHR_COL_WIDTH;
	CHECK(procTableLayout(withShared).showShared);
	CHECK_FALSE(procTableLayout(withShared - 1).showShared);
	const int withGpuMem = withShared + OPTIONAL_COL_WIDTH;
	CHECK(procTableLayout(withGpuMem).showGpuMemPercent);
	CHECK_FALSE(procTableLayout(withGpuMem - 1).showGpuMemPercent);
	CHECK(procTableLayout(withGpuMem + OPTIONAL_COL_WIDTH).showCopy);
	CHECK_FALSE(procTableLayout(withGpuMem + OPTIONAL_COL_WIDTH).showRender);
	CHECK(procTableLayout(withGpuMem + 2 * OPTIONAL_COL_WIDTH).showRender);

	const ProcTableLayout standard = procTableLayout(80);
	CHECK(standard == ProcTableLayout{.showRender = false,
									  .showCopy = false,
									  .showGpuMemPercent = false,
									  .showShared = false,
									  .showSystem = false,
									  .showRuntime = false,
									  .nameWidth = 80 - PROC_FIXED_WIDTH});

	CHECK(procTableLayout(40).nameWidth == MIN_NAME_WIDTH);
}

TEST_CASE("gpuMemPercent is the process's share of device memory, when the total is known")
{
	CHECK(*gpuMemPercent(331 * 1024, 12216.0) == doctest::Approx(2.71F).epsilon(0.01));
	CHECK(*gpuMemPercent(0, 12216.0) == doctest::Approx(0.0F));
	CHECK_FALSE(gpuMemPercent(1024, std::nullopt).has_value());
	CHECK_FALSE(gpuMemPercent(1024, 0.0).has_value());
}

TEST_CASE("smoothProcessList passes the first sample, blends the next, forgets departed pids")
{
	std::unordered_map<uint32_t, SmoothedUtil> state;

	std::vector<psInfo> procs{makeProc(1, 0, 10.0F), makeProc(2, 0, 20.0F)};
	smoothProcessList(procs, state);
	CHECK(procs[0].util.compute == doctest::Approx(10.0F));
	CHECK(procs[1].util.compute == doctest::Approx(20.0F));

	procs = {makeProc(1, 0, 20.0F)};
	smoothProcessList(procs, state);
	CHECK(*procs[0].util.compute == doctest::Approx(0.7F * 20.0F + 0.3F * 10.0F));
	CHECK(state.size() == 1);
	CHECK(state.contains(1));
}

TEST_CASE("calcLayout spaces the bars when it can, then goes dense, then compact")
{
	const LayoutDims roomy = calcLayout(40, 120, 1, false);
	CHECK(roomy.rowGap == 1);
	CHECK(roomy.devPanelH == BASE_PANEL_ROWS + BASE_BARS - 1);
	CHECK(roomy.procH >= 1);

	const LayoutDims withEu = calcLayout(60, 120, 1, true);
	CHECK(withEu.rowGap == 1);
	CHECK(withEu.devPanelH == BASE_PANEL_ROWS + EU_PANEL_ROWS + BASE_BARS + EU_BARS - 1);

	// Too short for the gaps: the same panel without them.
	const LayoutDims dense = calcLayout(24, 120, 1, false);
	CHECK(dense.rowGap == 0);
	CHECK(dense.devPanelH == BASE_PANEL_ROWS);

	const LayoutDims crowded = calcLayout(24, 80, 4, false);
	CHECK(crowded.rowGap == 0);
	CHECK(crowded.devPanelH == BASE_COMPACT_ROWS);
	CHECK(crowded.procH >= 1);

	CHECK(calcLayout(5, 80, 8, true).procH == 1);
}

TEST_CASE("formatRate fits a byte rate in the value column")
{
	CHECK(formatRate(0.0) == "0B/s");
	CHECK(formatRate(-5.0) == "0B/s");
	CHECK(formatRate(512.0) == "512B/s");
	CHECK(formatRate(58.34e6) == "58.3M/s");
	CHECK(formatRate(590.0e6) == "590M/s");
	CHECK(formatRate(12.34e9) == "12.3G/s");
	CHECK(formatRate(456.0e9) == "456G/s");
	CHECK(formatRate(999.99e6) == "1.0G/s");
	for (const double v : {1.0, 999.9, 99.96e3, 123.4e6, 999.4e9, 5.0e15}) {
		CAPTURE(v);
		CHECK(formatRate(v).size() <= static_cast<std::size_t>(VALUE_WIDTH));
	}
}

TEST_CASE("bandwidthScale only grows, starting from the reported peak")
{
	CHECK(bandwidthScale(0.0, 0, std::nullopt, std::nullopt) == doctest::Approx(0.0));
	CHECK(bandwidthScale(0.0, 59'000'000'000ULL, 1.0e9, 2.0e9) == doctest::Approx(59.0e9));
	// An understated peak gives way to what was measured, and the scale then stays.
	CHECK(bandwidthScale(59.0e9, 59'000'000'000ULL, 300.0e9, 1.0e9) == doctest::Approx(300.0e9));
	CHECK(bandwidthScale(300.0e9, 59'000'000'000ULL, 1.0e9, std::nullopt) == doctest::Approx(300.0e9));
}

TEST_CASE("calcLayout caps the bar and gives the rest of a wide screen to the graph")
{
	const LayoutDims wide = calcLayout(40, 240, 1, false);
	CHECK(wide.barWidth == MAX_BAR_WIDTH);
	CHECK(wide.sparkWidth == static_cast<int>(HISTORY_SAMPLES));
	CHECK(wide.sparkCol() + wide.sparkWidth <= wide.cols);

	const LayoutDims standard = calcLayout(40, 80, 1, false);
	CHECK(standard.barWidth >= MIN_BAR_WIDTH);
	CHECK(standard.barWidth <= MAX_BAR_WIDTH);
	CHECK(standard.sparkWidth >= MIN_SPARK_WIDTH);
	CHECK(standard.sparkCol() + standard.sparkWidth < standard.cols);

	// Too narrow for a graph: the bar and percentage still fit and the graph is dropped.
	const LayoutDims narrow = calcLayout(40, 50, 1, false);
	CHECK(narrow.sparkWidth == 0);
	CHECK(narrow.valueCol() + VALUE_WIDTH <= narrow.cols);
}

TEST_CASE("topInputs drops EU unless it is held")
{
	using metrics::MetricInput;
	const MetricInput without = topInputs(false);
	CHECK_FALSE(hasAny(without, MetricInput::EU));
	CHECK(hasAny(without, MetricInput::ENGINE));
	CHECK(hasAny(without, MetricInput::FDINFO));

	CHECK(hasAny(topInputs(true), MetricInput::EU));
}

TEST_CASE("runtimeLabel puts the APIs for the engines in use first")
{
	runtime::ProcessRuntime player;
	player.apis =
		(1U << static_cast<unsigned>(runtime::Api::Vulkan)) | (1U << static_cast<unsigned>(runtime::Api::VaApi));
	player.frameworks = 1U << static_cast<unsigned>(runtime::Framework::FFmpeg);
	CHECK(runtimeLabel(player, 0) == "FFmpeg/VK+VA");
	CHECK(runtimeLabel(player, ZES_ENGINE_TYPE_FLAG_MEDIA) == "FFmpeg/VA+VK");

	runtime::ProcessRuntime torch;
	torch.apis = (1U << static_cast<unsigned>(runtime::Api::LevelZero)) |
				 (1U << static_cast<unsigned>(runtime::Api::Sycl)) |
				 (1U << static_cast<unsigned>(runtime::Api::OpenGl));
	torch.frameworks = 1U << static_cast<unsigned>(runtime::Framework::PyTorch);
	CHECK(runtimeLabel(torch, ZES_ENGINE_TYPE_FLAG_RENDER) == "Torch/GL+SYCL+L0");
	CHECK(runtimeLabel(torch, ZES_ENGINE_TYPE_FLAG_COMPUTE) == "Torch/SYCL+L0+GL");

	CHECK(runtimeLabel({}, ZES_ENGINE_TYPE_FLAG_COMPUTE) == "-");
	runtime::ProcessRuntime onlyFramework;
	onlyFramework.frameworks = 1U << static_cast<unsigned>(runtime::Framework::OpenVino);
	CHECK(runtimeLabel(onlyFramework, 0) == "OV");
}

namespace {

constexpr uint64_t MiB = 1024ULL * 1024ULL;

} // namespace

TEST_CASE("sortMergedList by memory uses the attributed figure, and memSize where there is none")
{
	const std::vector<std::vector<psInfo>> lists{{makeProc(1, 900 * 1024), makeProc(2, 100 * 1024)}};
	// pid 1's memSize counts buffers it shares; only 50 MiB is its own.
	const std::vector<fdinfo::ProcessMemoryMap> memory{
		{{1, fdinfo::ProcessMemory{.gpu = 50 * MiB, .shared = 850 * MiB, .system = std::nullopt}}}};
	auto merged = buildMergedList(lists, {}, memory);
	REQUIRE(merged[0].mem.has_value());
	CHECK_FALSE(merged[1].mem.has_value());
	sortMergedList(merged, SortBy::Memory);
	CHECK(merged[0].proc->processId == 2);
	CHECK(merged[0].gpuMemKiB() == 100 * 1024);
	CHECK(merged[1].gpuMemKiB() == 50 * 1024);
}
