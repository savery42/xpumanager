/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Pure logic behind `xpu-smi top`: sparklines, process merging and sorting,
 * smoothing, layout and value formatting. No I/O, so unit tests can reach it.
 */

#ifndef IAL_TOP_MODEL_H
#define IAL_TOP_MODEL_H

#include "cmd_ps.h"
#include "metrics_registry.h"
#include "utility/compat/format.h"
#include "utility/units/units.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace top {

// UTF-8 block elements U+2581..U+2588 (1/8 to 8/8 height).
inline constexpr std::array<std::string_view, 9> SPARK_BLOCKS = {" ",
																 "\xe2\x96\x81",
																 "\xe2\x96\x82",
																 "\xe2\x96\x83",
																 "\xe2\x96\x84",
																 "\xe2\x96\x85",
																 "\xe2\x96\x86",
																 "\xe2\x96\x87",
																 "\xe2\x96\x88"};

inline constexpr float EWMA_ALPHA = 0.7F;

template <std::floating_point T, std::size_t N> class SparklineBuffer
{
	std::array<T, N> samples{};
	std::size_t next = 0;
	std::size_t filled = 0;

public:
	void push(T val) noexcept
	{
		samples[next] = val;
		next = (next + 1) % N;
		filled = std::min(filled + 1, N);
	}

	[[nodiscard]] std::vector<T> last(std::size_t width) const
	{
		const std::size_t take = std::min(width, filled);
		std::vector<T> out(take);
		for (std::size_t i = 0; i < take; ++i) {
			out[i] = samples[(next + N - take + i) % N];
		}
		return out;
	}

	[[nodiscard]] bool empty() const noexcept { return filled == 0; }
	[[nodiscard]] std::size_t count() const noexcept { return filled; }
};

/// Samples a sparkline remembers; also the widest one worth drawing.
inline constexpr std::size_t HISTORY_SAMPLES = 120;
using History = SparklineBuffer<double, HISTORY_SAMPLES>;

/**
 * @brief Which SPARK_BLOCKS glyph draws @p value.
 *
 * A sample that exists is never blank: an idle GPU draws the lowest block, so it reads as a
 * flat baseline, while a column with no sample yet stays empty. The tallest glyph stops one
 * eighth short of the cell, so full-load graphs on adjacent rows do not fuse into one slab.
 *
 * @param[in] value  The sample.
 * @param[in] peak   The value that reaches the tallest glyph.
 *
 * @return An index into SPARK_BLOCKS, 1 (lowest block) to 7 (seven eighths).
 */
[[nodiscard]] inline std::size_t sparkLevel(double value, double peak = 100.0) noexcept
{
	return static_cast<std::size_t>(1.0 + std::clamp(value / peak, 0.0, 1.0) * 6.0 + 0.5);
}

/// A horizontal bar in eighths of a cell: @c full whole cells, then one cell filled @c eighths/8.
struct BarFill
{
	int full = 0;
	int eighths = 0; ///< 0 (no partial cell) to 7

	bool operator==(const BarFill &) const = default;
};

/**
 * @brief How much of a @p width -cell bar @p pct fills.
 *
 * @param[in] pct    Percent, clamped to 0-100.
 * @param[in] width  Bar width in cells.
 */
[[nodiscard]] inline BarFill barFill(double pct, int width) noexcept
{
	if (width <= 0) {
		return {};
	}
	const int total = static_cast<int>(std::clamp(pct, 0.0, 100.0) / 100.0 * static_cast<double>(width) * 8.0 + 0.5);
	return {.full = total / 8, .eighths = total % 8};
}

/// UTF-8 left blocks U+258F..U+2589 (1/8 to 7/8 wide), indexed by BarFill::eighths - 1.
inline constexpr std::array<std::string_view, 7> BAR_PARTIAL = {
	"\xe2\x96\x8f", "\xe2\x96\x8e", "\xe2\x96\x8d", "\xe2\x96\x8c", "\xe2\x96\x8b", "\xe2\x96\x8a", "\xe2\x96\x89"};

enum class SortBy
{
	Memory, ///< GPU memory
	Compute,
	Cpu,
	Pid
};

/**
 * @brief Maps a --sort argument to a key.
 *
 * @param[in] s  "mem", "compute", "cpu" or "pid".
 *
 * @retval key      @p s names a sort key.
 * @retval nullopt  Anything else.
 */
[[nodiscard]] inline std::optional<SortBy> parseSortKey(std::string_view s) noexcept
{
	if (s == "mem") {
		return SortBy::Memory;
	}
	if (s == "compute") {
		return SortBy::Compute;
	}
	if (s == "cpu") {
		return SortBy::Cpu;
	}
	if (s == "pid") {
		return SortBy::Pid;
	}
	return std::nullopt;
}

/**
 * @brief Parses a whole string as a number.
 *
 * @retval value    @p s is entirely a number.
 * @retval nullopt  @p s is empty or has anything left over, e.g. "N/A" or "12x".
 */
[[nodiscard]] inline std::optional<double> parseNumber(std::string_view s) noexcept
{
	if (s.empty()) {
		return std::nullopt;
	}
	double val = 0.0;
	const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), val);
	if (ec != std::errc{} || ptr != s.data() + s.size()) {
		return std::nullopt;
	}
	return val;
}

/// One reading of a process's accumulated CPU time and when it was taken.
struct HostReading
{
	std::chrono::nanoseconds cpuTime{};
	std::chrono::steady_clock::time_point at{};
};

/**
 * @brief CPU use between two readings of one process, where 100% is one core busy.
 *
 * A multithreaded process can exceed 100%.
 *
 * @retval percent  @p after is later than @p before and its CPU time did not go backwards.
 * @retval nullopt  Otherwise: no time passed, or the pid now belongs to a different process.
 */
[[nodiscard]] inline std::optional<float> cpuPercent(const HostReading &before, const HostReading &after) noexcept
{
	if (after.at <= before.at || after.cpuTime < before.cpuTime) {
		return std::nullopt;
	}
	const auto wall = std::chrono::duration<double>(after.at - before.at).count();
	const auto busy = std::chrono::duration<double>(after.cpuTime - before.cpuTime).count();
	return static_cast<float>(busy / wall * 100.0);
}

/// The host side of a process, as top shows it.
struct HostUsage
{
	std::optional<float> cpu;					///< see cpuPercent()
	std::optional<xpum::units::Bytes> resident; ///< resident set size
};

using HostUsageMap = std::unordered_map<uint32_t, HostUsage>;

struct MergedProc
{
	int gpuIdx;
	const psInfo *proc;
	HostUsage host{};
};

/// Every process row across GPUs; a process on two GPUs is two rows sharing its host usage.
[[nodiscard]] inline std::vector<MergedProc> buildMergedList(const std::vector<std::vector<psInfo>> &lists,
															 const HostUsageMap &host = {})
{
	std::vector<MergedProc> out;
	for (int i = 0; i < static_cast<int>(lists.size()); ++i) {
		for (const auto &p : lists[static_cast<std::size_t>(i)]) {
			const auto it = host.find(p.processId);
			out.push_back({i, &p, it != host.end() ? it->second : HostUsage{}});
		}
	}
	return out;
}

// Descending keys put processes with no figure last: an empty optional compares below any value.
inline void sortMergedList(std::vector<MergedProc> &list, SortBy sortBy)
{
	switch (sortBy) {
	case SortBy::Memory:
		std::ranges::sort(list,
						  [](const MergedProc &a, const MergedProc &b) { return a.proc->memSize > b.proc->memSize; });
		break;
	case SortBy::Compute:
		std::ranges::sort(
			list, [](const MergedProc &a, const MergedProc &b) { return a.proc->util.compute > b.proc->util.compute; });
		break;
	case SortBy::Cpu:
		std::ranges::sort(list, [](const MergedProc &a, const MergedProc &b) { return a.host.cpu > b.host.cpu; });
		break;
	case SortBy::Pid:
		std::ranges::sort(
			list, [](const MergedProc &a, const MergedProc &b) { return a.proc->processId < b.proc->processId; });
		break;
	}
}

// Process table columns, in display order. Name is last and takes what is left of the row.
// clang-format off
inline constexpr int PROC_INDENT     = 2;
inline constexpr int PROC_GAP        = 2;
inline constexpr int COL_PID         = 7;
inline constexpr int COL_GPU         = 3;
inline constexpr int COL_TYPE        = 7;
inline constexpr int COL_GPU_MEM     = 7;
inline constexpr int COL_PERCENT     = 6;
inline constexpr int COL_HOST_MEM    = 7;
inline constexpr int MIN_NAME_WIDTH  = 8;
// clang-format on

/// PID, GPU, Type, GPU memory, Cmp%, Med%, CPU%, host memory, each with the gap after it.
inline constexpr int PROC_FIXED_WIDTH =
	PROC_INDENT + COL_PID + COL_GPU + COL_TYPE + COL_GPU_MEM + 3 * COL_PERCENT + COL_HOST_MEM + 8 * PROC_GAP;
/// A percentage column the table can drop: Rnd%, Cpy% or GMem%.
inline constexpr int OPTIONAL_COL_WIDTH = COL_PERCENT + PROC_GAP;

struct ProcTableLayout
{
	bool showRender = true;
	bool showCopy = true;
	bool showGpuMemPercent = true;
	int nameWidth = MIN_NAME_WIDTH;

	bool operator==(const ProcTableLayout &) const = default;
};

/**
 * @brief Which process columns fit in @p cols.
 *
 * Optional columns go in this order until the name has MIN_NAME_WIDTH: Rnd% and Cpy%, which
 * are usually idle, then GMem%, which GPUMiB already approximates. On a terminal too narrow
 * even then, the name keeps MIN_NAME_WIDTH and the row overflows.
 */
[[nodiscard]] inline ProcTableLayout procTableLayout(int cols) noexcept
{
	ProcTableLayout table;
	const std::array<bool ProcTableLayout::*, 3> dropOrder{&ProcTableLayout::showRender, &ProcTableLayout::showCopy,
														   &ProcTableLayout::showGpuMemPercent};
	int room = cols - PROC_FIXED_WIDTH - static_cast<int>(dropOrder.size()) * OPTIONAL_COL_WIDTH;
	for (const auto column : dropOrder) {
		if (room >= MIN_NAME_WIDTH) {
			break;
		}
		table.*column = false;
		room += OPTIONAL_COL_WIDTH;
	}
	table.nameWidth = std::max(MIN_NAME_WIDTH, room);
	return table;
}

/**
 * @brief A process's GPU memory as a share of its device's.
 *
 * @param[in] usedKiB   psInfo::memSize.
 * @param[in] totalMiB  The device's memory.total.
 *
 * @retval percent  @p totalMiB is known and positive.
 * @retval nullopt  Otherwise.
 */
[[nodiscard]] inline std::optional<float> gpuMemPercent(uint64_t usedKiB, std::optional<double> totalMiB) noexcept
{
	if (!totalMiB || *totalMiB <= 0.0) {
		return std::nullopt;
	}
	return static_cast<float>(static_cast<double>(usedKiB) / 1024.0 / *totalMiB * 100.0);
}

struct SmoothedUtil
{
	std::optional<float> compute;
	std::optional<float> copy;
	std::optional<float> render;
	std::optional<float> media;
};

/// Exponentially smooths each process's engine figures in place; forgets pids that left @p procs.
inline void smoothProcessList(std::vector<psInfo> &procs, std::unordered_map<uint32_t, SmoothedUtil> &state)
{
	const auto ema = [](std::optional<float> prev, std::optional<float> curr) -> std::optional<float> {
		if (!curr) {
			return prev;
		}
		if (!prev) {
			return curr;
		}
		return EWMA_ALPHA * *curr + (1.0F - EWMA_ALPHA) * *prev;
	};

	for (auto &p : procs) {
		auto &s = state[p.processId];
		s.compute = ema(s.compute, p.util.compute);
		s.copy = ema(s.copy, p.util.copy);
		s.render = ema(s.render, p.util.render);
		s.media = ema(s.media, p.util.media);
		p.util.compute = s.compute;
		p.util.copy = s.copy;
		p.util.render = s.render;
		p.util.media = s.media;
	}
	std::erase_if(state, [&procs](const auto &kv) {
		return std::ranges::none_of(procs, [&](const psInfo &p) { return p.processId == kv.first; });
	});
}

inline constexpr int VALUE_WIDTH = 7; ///< "100.0%" or a rate such as "12.3G/s"

struct LayoutDims
{
	int rows = 0;
	int cols = 0;
	int devPanelH = 0;
	int procTop = 0;
	int procH = 0;
	int barWidth = 0;
	int sparkWidth = 0; ///< 0 when there is no room for a graph
	int labelWidth = 18;
	int rowGap = 0; ///< blank rows between bar rows

	/// Column of a bar row's '[': the 2-space indent, the label, and one space.
	[[nodiscard]] int barCol() const noexcept { return labelWidth + 3; }
	/// Column of the value: past the bar's cells and its closing ']' and a space.
	[[nodiscard]] int valueCol() const noexcept { return barCol() + 1 + barWidth + 2; }
	/// Column of the history graph: past the value and two spaces.
	[[nodiscard]] int sparkCol() const noexcept { return valueCol() + VALUE_WIDTH + 2; }
};

inline constexpr int BASE_PANEL_ROWS = 12;
inline constexpr int BASE_BARS = 8; ///< GPU Util, Compute, Render, Media, Copy, Memory, Mem Read, Mem Write
inline constexpr int EU_BARS = 3;
inline constexpr int EU_PANEL_ROWS = 3;
inline constexpr int BASE_COMPACT_ROWS = 5;
inline constexpr int EU_COMPACT_ROWS = 1;
inline constexpr int MIN_BAR_WIDTH = 10;
inline constexpr int MAX_BAR_WIDTH = 32;
inline constexpr int MIN_SPARK_WIDTH = 10;
inline constexpr int FULL_PANEL_MARGIN_ROWS = 9; ///< header, process header, footer and a few process rows

[[nodiscard]] inline LayoutDims calcLayout(int rows, int cols, int numDevices, bool euAvail)
{
	LayoutDims d;
	d.rows = rows;
	d.cols = cols;

	const int fullH = BASE_PANEL_ROWS + (euAvail ? EU_PANEL_ROWS : 0);
	const int compactH = BASE_COMPACT_ROWS + (euAvail ? EU_COMPACT_ROWS : 0);
	// One blank row between consecutive bars keeps neighbouring rows from running together; it is
	// the first thing to go when the terminal is short.
	const int spacedH = fullH + BASE_BARS + (euAvail ? EU_BARS : 0) - 1;
	if (numDevices * spacedH + FULL_PANEL_MARGIN_ROWS <= rows) {
		d.devPanelH = spacedH;
		d.rowGap = 1;
	} else if (numDevices * fullH + FULL_PANEL_MARGIN_ROWS <= rows) {
		d.devPanelH = fullH;
	} else {
		d.devPanelH = compactH;
	}
	d.procTop = 1 + numDevices * d.devPanelH + 1;
	d.procH = std::max(1, rows - d.procTop - 2);

	// Bars stay a readable length on wide terminals; the rest of the width goes to the graph.
	const int room = cols - d.barCol() - 1 - 2 - VALUE_WIDTH - 2 - 1;
	d.barWidth = std::clamp(room * 2 / 5, MIN_BAR_WIDTH, MAX_BAR_WIDTH);
	const int sparkRoom = cols - d.sparkCol() - 1;
	d.sparkWidth = sparkRoom >= MIN_SPARK_WIDTH ? std::min(sparkRoom, static_cast<int>(HISTORY_SAMPLES)) : 0;
	return d;
}

/**
 * @brief A byte rate in at most VALUE_WIDTH characters, decimal units.
 *
 * @retval "590M/s"   Three significant digits from 100 of a unit up.
 * @retval "12.3G/s"  One decimal below 100 of a unit.
 * @retval "0B/s"     Zero or negative.
 */
[[nodiscard]] inline std::string formatRate(double bytesPerSecond)
{
	constexpr std::array<std::string_view, 5> units{"B", "K", "M", "G", "T"};
	double v = std::max(0.0, bytesPerSecond);
	std::size_t unit = 0;
	// 999.5 rounds to "1000", one character too many, so it moves up a unit as well.
	while (v >= 999.5 && unit + 1 < units.size()) {
		v /= 1000.0;
		++unit;
	}
	if (unit == 0 || v >= 99.95) {
		return xpum::compat::format("{:.0f}{}/s", v, units[unit]);
	}
	return xpum::compat::format("{:.1f}{}/s", v, units[unit]);
}

/**
 * @brief The full-scale value for a GPU's memory read and write bars.
 *
 * The larger of the device's reported peak and everything seen so far: a peak the driver
 * understates (or does not report) can then never pin the bars at 100%.
 *
 * @param[in] previous   The scale from the last frame (0 at first).
 * @param[in] peak       MetricCache::memMaxBandwidth in bytes/s, 0 when unknown.
 * @param[in] readRate   This frame's read rate in bytes/s.
 * @param[in] writeRate  This frame's write rate in bytes/s.
 */
[[nodiscard]] inline double bandwidthScale(double previous, uint64_t peak, std::optional<double> readRate,
										   std::optional<double> writeRate) noexcept
{
	return std::max({previous, static_cast<double>(peak), readRate.value_or(0.0), writeRate.value_or(0.0)});
}

/// Every registry field top displays; also what decides its sampling inputs.
inline constexpr std::array<std::string_view, 22> TOP_FIELDS{
	"name",
	"pci.bus_id",
	"utilization.gpu",
	"utilization.compute",
	"utilization.render",
	"utilization.media",
	"utilization.copy",
	"utilization.memory",
	"memory.used",
	"memory.total",
	"memory.read.bandwidth",
	"memory.write.bandwidth",
	"temperature.gpu",
	"power.draw",
	"power.limit",
	"fan.speed",
	"clocks.current.graphics",
	"clocks.throttle.reason",
	"pcie.tx.throughput",
	"pcie.rx.throughput",
	"pcie.link.gen.current",
	"pcie.link.width.current",
};

/**
 * @brief The raw sampling one top frame needs.
 *
 * @param[in] euHeld  Whether the EU feature was admitted (a core device handle exists).
 *
 * @return The union of TOP_FIELDS' MetricInput, with EU added when @p euHeld and
 *         removed otherwise.
 */
[[nodiscard]] inline metrics::MetricInput topInputs(bool euHeld)
{
	auto inputs = metrics::MetricInput::NONE;
	for (const std::string_view name : TOP_FIELDS) {
		if (const auto qm = metrics::findMetric(name); qm) {
			inputs = inputs | qm->inputs;
		}
	}
	return euHeld ? (inputs | metrics::MetricInput::EU) : (inputs & ~metrics::MetricInput::EU);
}

/**
 * @brief Formats a registry value for the panel.
 *
 * @param[in] raw   The getter's output.
 * @param[in] unit  The field's QueryMetric::unit.
 *
 * @retval ""     @p raw is empty.
 * @retval raw    @p raw is not a number (e.g. a throttle reason list).
 * @retval "N/A"  A negative number: the driver's "unknown" (PCIe link -1).
 * @retval text   Otherwise, the number with its unit.
 */
[[nodiscard]] inline std::string formatField(std::string_view raw, std::string_view unit)
{
	if (raw.empty()) {
		return {};
	}
	const auto parsed = parseNumber(raw);
	if (!parsed) {
		return std::string{raw};
	}
	const double v = *parsed;
	if (v < 0.0) {
		return "N/A";
	}
	if (unit == "%") {
		return xpum::compat::format("{:.1f}%", v);
	}
	if (unit == "C") {
		return xpum::compat::format("{:.0f} C", v);
	}
	if (unit == "W") {
		return xpum::compat::format("{:.1f} W", v);
	}
	if (unit == "MHz") {
		return xpum::compat::format("{:.0f} MHz", v);
	}
	if (unit == "MiB") {
		return xpum::compat::format("{:.0f} MiB", v);
	}
	if (unit == "kB/s") {
		return xpum::compat::format("{:.1f} MB/s", v / 1000.0);
	}
	if (unit == "MB/s") {
		return xpum::compat::format("{:.1f} MB/s", v);
	}
	if (unit.empty()) {
		return xpum::compat::format("{:.0f}", v);
	}
	return xpum::compat::format("{:.1f} {}", v, unit);
}

} // namespace top

#endif // IAL_TOP_MODEL_H
