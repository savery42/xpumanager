/*
 * Copyright (C) 2025-2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 */

#include "cmd_top.h"
#include "ansi_term.h"
#include "cmd_ps.h"
#include "cmds.h"
#include "debug.h"
#include "keyboard.h"
#include "local_time.h"
#include "metrics_registry.h"
#include "proc_fdinfo.h"
#include "process_control.h"
#include "process_stats.h"
#include "requirements.h"
#include "table_builder.h"
#include "terminal.h"
#include "top_model.h"
#include "utility/compat/format.h"

#include <CLI/CLI.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <os.h>

namespace {

using namespace top;

// Color-pair IDs (mapped to ANSI fg colors in cpToFg())
constexpr int CP_HEADER = 1;
constexpr int CP_BAR_LOW = 2;
constexpr int CP_BAR_MED = 3;
constexpr int CP_BAR_HIGH = 4;
constexpr int CP_EU = 5;
constexpr int CP_LOAD = 6; // green/yellow/red by the value shown
constexpr int CP_MEM = 7;
constexpr int CP_DIM = 8;

constexpr auto KEY_POLL = std::chrono::milliseconds{50};
constexpr unsigned char CTRL_C = 0x03;

std::atomic<bool> g_quit{false}; // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)
bool g_colorsEnabled = false;	 // NOLINT(cppcoreguidelines-avoid-non-const-global-variables)

// EU bars are top's only compute-runtime (zet) consumer, so they alone raise the driver to Full.
constexpr std::array TOP_FEATURES{
	OptionalFeature{"EU active/stall/idle bars", Capability::PerfObserve, DriverMode::Full}};

struct DeviceHistory
{
	History gpuUtil;
	History compute;
	History renderUtil;
	History media;
	History copyUtil;
	History euActive;
	History euStall;
	History euIdle;
	History memUtil;
	History memRead;			 ///< bytes/s
	History memWrite;			 ///< bytes/s
	double bandwidthScale = 0.0; ///< full scale of both, bytes/s; see bandwidthScale()
};

struct TopState
{
	std::vector<metrics::MetricCache> caches;
	std::vector<DeviceHistory> history;
	std::vector<std::vector<psInfo>> processLists;
	std::vector<std::unordered_map<uint32_t, SmoothedUtil>> smoothed;
	// The walk that closed the last window opens the next: one /proc walk per tick serves every
	// device's utilization and process list.
	fdinfo::Walk lastWalk;
	// Each device's memory.total in MiB, for GMem%; read once, it does not change.
	std::vector<std::optional<double>> gpuMemTotalMiB;
	// Host CPU and memory per pid, shared by every GPU the process appears on.
	std::unordered_map<uint32_t, HostReading> hostReadings;
	HostUsageMap host;

	int selectedRow = 0;
	int scrollOffset = 0;
	LayoutDims layout;
	TerminalSize screen{24, 80};

	// Outcome of the last kill, shown in the footer until the next key press.
	std::string status;
	bool statusFailed = false;
};

class StopSignals
{
public:
	StopSignals() : previousInterrupt(install(SIGINT)), previousTerminate(install(SIGTERM)) {}
	~StopSignals()
	{
		std::signal(SIGINT, previousInterrupt);
		std::signal(SIGTERM, previousTerminate);
	}
	StopSignals(const StopSignals &) = delete;
	StopSignals &operator=(const StopSignals &) = delete;
	StopSignals(StopSignals &&) = delete;
	StopSignals &operator=(StopSignals &&) = delete;

private:
	using Handler = void (*)(int);

	static void onStop(int /*sig*/) { g_quit.store(true, std::memory_order_relaxed); }

	// A failed std::signal leaves the default disposition, which is what to restore.
	static Handler install(int sig)
	{
		const Handler previous = std::signal(sig, onStop);
		return previous == SIG_ERR ? SIG_DFL : previous;
	}

	Handler previousInterrupt;
	Handler previousTerminate;
};

bool shouldUseColor(bool noColor) noexcept
{
	if (noColor || !stdoutIsTerminal()) {
		return false;
	}
	const char *envTerm = std::getenv("TERM");
	return !(envTerm != nullptr && std::string_view{envTerm} == "dumb");
}

// Map CP_* color-pair IDs to ANSI foreground color codes.
int cpToFg(int colorPair) noexcept
{
	switch (colorPair) {
	case CP_BAR_LOW:
		return term::FG_GREEN;
	case CP_BAR_MED:
		return term::FG_YELLOW;
	case CP_BAR_HIGH:
		return term::FG_RED;
	case CP_EU:
		return term::FG_CYAN;
	case CP_MEM:
		return term::FG_MAGENTA;
	default:
		return term::FG_DEFAULT;
	}
}

int barColorPair(double pct) noexcept
{
	if (pct >= 90.0) {
		return CP_BAR_HIGH;
	}
	if (pct >= 70.0) {
		return CP_BAR_MED;
	}
	return CP_BAR_LOW;
}

// Foreground for a cell of a bar or graph that stands for @p pct.
int fgAt(int colorPair, double pct) noexcept { return cpToFg(colorPair == CP_LOAD ? barColorPair(pct) : colorPair); }

struct FieldValue
{
	std::string raw;
	std::string_view unit;
};

[[nodiscard]] std::optional<FieldValue> queryField(devInfo &dev, std::string_view name,
												   const metrics::MetricCache &cache)
{
	const auto qm = metrics::findMetric(name);
	if (!qm) {
		return std::nullopt;
	}
	metrics::MetricValue mv;
	if (qm->getter(dev, mv, cache) != ZE_RESULT_SUCCESS) {
		return std::nullopt;
	}
	return FieldValue{std::move(mv), qm->unit};
}

/// The field formatted for display; empty when the registry has no value for it.
[[nodiscard]] std::string display(devInfo &dev, std::string_view name, const metrics::MetricCache &cache)
{
	const auto field = queryField(dev, name, cache);
	return field ? formatField(field->raw, field->unit) : std::string{};
}

/// The field as a number; nullopt when it is absent, not numeric or negative (the driver's "unknown").
[[nodiscard]] std::optional<double> percent(devInfo &dev, std::string_view name, const metrics::MetricCache &cache)
{
	const auto field = queryField(dev, name, cache);
	if (!field) {
		return std::nullopt;
	}
	const auto value = parseNumber(field->raw);
	if (!value || *value < 0.0) {
		return std::nullopt;
	}
	return value;
}

[[nodiscard]] std::string orNotAvailable(std::string text) { return text.empty() ? std::string{"N/A"} : text; }

[[nodiscard]] bool hasText(const std::string &text) noexcept { return !text.empty() && text != "N/A"; }

// One row: label, a bar whose cells are coloured by where they sit on the scale (so a full bar
// runs green to red), @p text, then the recent history as a coloured one-row graph. @p value and
// the history are in the units of @p full.
int drawGaugeRow(int row, std::string_view label, std::optional<double> value, double full, std::string_view text,
				 History &history, const LayoutDims &layout, int colorPair, bool compact)
{
	const int bw = layout.barWidth;
	const auto share = [full](double v) { return full > 0.0 ? 100.0 * v / full : 0.0; };

	// Label - always bold for readability even in no-color mode
	term::gotoPos(row, 0);
	term::attrBold();
	std::fprintf(stdout, "  %*.*s", layout.labelWidth, static_cast<int>(label.size()), label.data());
	term::attrReset();

	term::gotoPos(row, layout.barCol());
	std::putchar('[');

	const BarFill fill = value ? barFill(share(*value), bw) : BarFill{};
	int cell = 0;
	int lastFg = -1;
	const auto paint = [&](std::string_view glyph) {
		if (g_colorsEnabled) {
			const int fg = fgAt(colorPair, 100.0 * static_cast<double>(cell + 1) / static_cast<double>(bw));
			if (fg != lastFg) {
				term::setFg(fg);
				lastFg = fg;
			}
		}
		std::fwrite(glyph.data(), 1, glyph.size(), stdout);
		++cell;
	};
	for (int i = 0; i < fill.full; ++i) {
		paint(SPARK_BLOCKS[8]);
	}
	if (fill.eighths > 0) {
		paint(BAR_PARTIAL[static_cast<std::size_t>(fill.eighths - 1)]);
	}
	if (g_colorsEnabled) {
		term::attrReset();
		term::attrDim();
	}
	for (; cell < bw; ++cell) {
		std::fputs("\xe2\x94\x80", stdout); // thin line: the empty track, clear of the rows around it
	}
	term::attrReset();
	std::putchar(']');

	term::gotoPos(row, layout.valueCol());
	std::fprintf(stdout, "%*.*s", VALUE_WIDTH, static_cast<int>(text.size()), text.data());

	// Graph (full mode only): oldest sample first, so it starts beside the percentage and scrolls
	// once it fills the field. Cells with no sample yet show a faint dot to mark the extent.
	if (!compact && layout.sparkWidth > 0) {
		if (value) {
			history.push(*value);
		}
		const auto samples = history.last(static_cast<std::size_t>(layout.sparkWidth));
		term::gotoPos(row, layout.sparkCol());
		lastFg = -1;
		for (const double v : samples) {
			if (g_colorsEnabled) {
				const int fg = fgAt(colorPair, share(v));
				if (fg != lastFg) {
					term::setFg(fg);
					lastFg = fg;
				}
			}
			std::fputs(SPARK_BLOCKS[sparkLevel(share(v))].data(), stdout);
		}
		term::attrReset();
		if (g_colorsEnabled) {
			term::attrDim();
		}
		for (int i = static_cast<int>(samples.size()); i < layout.sparkWidth; ++i) {
			std::fputs("\xc2\xb7", stdout);
		}
		term::attrReset();
	}

	term::attrReset();
	term::clearToEol();
	return row + 1;
}

int drawBarRow(int row, std::string_view label, std::optional<double> pct, History &history, const LayoutDims &layout,
			   int colorPair, bool compact)
{
	const std::string text = pct ? xpum::compat::format("{:.1f}%", *pct) : std::string{"N/A"};
	return drawGaugeRow(row, label, pct, 100.0, text, history, layout, colorPair, compact);
}

/// A rate field in bytes/s; the registry reports these two in kB/s.
[[nodiscard]] std::optional<double> bandwidth(devInfo &dev, std::string_view name, const metrics::MetricCache &cache)
{
	const auto kBps = percent(dev, name, cache);
	return kBps ? std::optional<double>{*kBps * 1000.0} : std::nullopt;
}

int renderDevicePanel(int startRow, devInfo &dev, const metrics::MetricCache &cache, DeviceHistory &hist,
					  const LayoutDims &layout)
{
	const bool compact = (layout.devPanelH <= BASE_COMPACT_ROWS + EU_COMPACT_ROWS);
	int row = startRow;

	const auto fieldText = [&](std::string_view name) {
		const auto field = queryField(dev, name, cache);
		return field ? field->raw : std::string{};
	};
	const std::string devName = fieldText("name");
	const std::string bdf = fieldText("pci.bus_id");

	term::gotoPos(row, 0);
	term::attrBold();
	if (g_colorsEnabled) {
		term::setFg(term::FG_WHITE);
	}
	std::fprintf(stdout, " GPU %u: %s  (%s)", dev.index, devName.empty() ? "Unknown" : devName.c_str(),
				 bdf.empty() ? "?" : bdf.c_str());
	term::clearToEol();
	term::attrReset();
	++row;

	// Blank rows are cleared, not skipped, so a layout change leaves no stale text behind.
	bool firstBar = true;
	const auto spaceBars = [&] {
		if (!firstBar) {
			for (int i = 0; i < layout.rowGap; ++i) {
				term::gotoPos(row++, 0);
				term::clearToEol();
			}
		}
		firstBar = false;
	};
	const auto bar = [&](std::string_view label, std::string_view field, History &history, int colorPair) {
		spaceBars();
		row = drawBarRow(row, label, percent(dev, field, cache), history, layout, colorPair, compact);
	};

	bar("GPU Util", "utilization.gpu", hist.gpuUtil, CP_LOAD);
	bar("Compute", "utilization.compute", hist.compute, CP_LOAD);
	if (!compact) {
		bar("Render", "utilization.render", hist.renderUtil, CP_LOAD);
		bar("Media", "utilization.media", hist.media, CP_LOAD);
		bar("Copy", "utilization.copy", hist.copyUtil, CP_LOAD);
	}

	{
		const std::string memUsed = display(dev, "memory.used", cache);
		const std::string memTotal = display(dev, "memory.total", cache);
		bar("Memory", "utilization.memory", hist.memUtil, CP_MEM);
		std::string detail;
		if (!memUsed.empty() && !memTotal.empty()) {
			detail = memUsed + " / " + memTotal;
		}
		const auto readRate = bandwidth(dev, "memory.read.bandwidth", cache);
		const auto writeRate = bandwidth(dev, "memory.write.bandwidth", cache);
		// Full mode puts the detail on the row BASE_PANEL_ROWS leaves for it, under the bar, and
		// charts read and write on their own rows; compact mode has no graph, so the detail and the
		// rates fit beside the percentage.
		if (!compact) {
			if (!detail.empty()) {
				term::gotoPos(row, layout.barCol() + 1);
				std::fputs(detail.c_str(), stdout);
				term::clearToEol();
				++row;
			}
			hist.bandwidthScale = bandwidthScale(hist.bandwidthScale, cache.memMaxBandwidth, readRate, writeRate);
			const auto rateRow = [&](std::string_view label, std::optional<double> rate, History &history) {
				spaceBars();
				row = drawGaugeRow(row, label, rate, hist.bandwidthScale, rate ? formatRate(*rate) : "N/A", history,
								   layout, CP_LOAD, compact);
			};
			rateRow("Mem Read", readRate, hist.memRead);
			rateRow("Mem Write", writeRate, hist.memWrite);
		} else {
			if (readRate || writeRate) {
				detail += xpum::compat::format("  Rd:{} Wr:{}", readRate ? formatRate(*readRate) : "N/A",
											   writeRate ? formatRate(*writeRate) : "N/A");
			}
			const int col = layout.sparkCol();
			const int room = layout.cols - col - 1;
			if (!detail.empty() && room > 0) {
				term::gotoPos(row - 1, col);
				std::fprintf(stdout, "%.*s", room, detail.c_str());
			}
		}
	}

	if (cache.euAvail) {
		// euActive/euStall/euIdle are in per-mille (0-1000); divide by EU_PERMILLE_SCALE to get %.
		const double actPct = static_cast<double>(cache.euSample.euActive) / EU_PERMILLE_SCALE;
		const double stallPct = static_cast<double>(cache.euSample.euStall) / EU_PERMILLE_SCALE;
		const double idlePct = static_cast<double>(cache.euSample.euIdle) / EU_PERMILLE_SCALE;
		if (!compact) {
			spaceBars();
			row = drawBarRow(row, "EU Active", actPct, hist.euActive, layout, CP_EU, compact);
			spaceBars();
			row = drawBarRow(row, "EU Stall", stallPct, hist.euStall, layout, CP_BAR_MED, compact);
			spaceBars();
			row = drawBarRow(row, "EU Idle", idlePct, hist.euIdle, layout, CP_DIM, compact);
		} else {
			term::gotoPos(row, 0);
			std::fprintf(stdout, "  EU  Active:%.1f%%  Stall:%.1f%%  Idle:%.1f%%", actPct, stallPct, idlePct);
			term::clearToEol();
			++row;
		}
	}

	{
		const std::string temp = display(dev, "temperature.gpu", cache);
		const std::string power = display(dev, "power.draw", cache);
		const std::string limit = display(dev, "power.limit", cache);
		const std::string fan = display(dev, "fan.speed", cache);
		const std::string freq = display(dev, "clocks.current.graphics", cache);
		const std::string throttle = fieldText("clocks.throttle.reason");
		const bool throttleActive = !throttle.empty() && throttle != "None" && throttle != "0";

		term::gotoPos(row, 0);
		if (hasText(temp)) {
			std::fprintf(stdout, "  Temp: %s", temp.c_str());
		}
		if (hasText(power)) {
			std::fprintf(stdout, "  Power: %s", power.c_str());
			if (hasText(limit)) {
				std::fprintf(stdout, "/%s", limit.c_str());
			}
		}
		if (hasText(fan)) {
			std::fprintf(stdout, "  Fan: %s", fan.c_str());
		}
		if (hasText(freq)) {
			std::fprintf(stdout, "  Freq: %s", freq.c_str());
		}
		if (throttleActive) {
			if (g_colorsEnabled) {
				term::setFg(term::FG_RED);
			}
			std::fprintf(stdout, "  [throttle: %s]", throttle.c_str());
			if (g_colorsEnabled) {
				term::attrReset();
			}
		}
		term::clearToEol();
		term::attrReset();
		++row;
	}

	if (!compact) {
		const std::string tx = display(dev, "pcie.tx.throughput", cache);
		const std::string rx = display(dev, "pcie.rx.throughput", cache);
		const auto gen = percent(dev, "pcie.link.gen.current", cache);
		const auto width = percent(dev, "pcie.link.width.current", cache);
		term::gotoPos(row, 0);
		std::fputs("  PCIe", stdout);
		if (gen && width) {
			std::fprintf(stdout, " Gen%dx%d", static_cast<int>(*gen), static_cast<int>(*width));
		}
		if (hasText(tx)) {
			std::fprintf(stdout, "  Tx: %s", tx.c_str());
		}
		if (hasText(rx)) {
			std::fprintf(stdout, "  Rx: %s", rx.c_str());
		}
		term::clearToEol();
		++row;
	}

	return row;
}

[[nodiscard]] std::optional<float> gpuMemPercentOf(const TopState &state, const MergedProc &mp)
{
	const auto idx = static_cast<std::size_t>(mp.gpuIdx);
	return gpuMemPercent(mp.proc->memSize,
						 idx < state.gpuMemTotalMiB.size() ? state.gpuMemTotalMiB[idx] : std::nullopt);
}

void renderProcessTable(const TopState &state, SortBy sortBy, int startRow, int procH, int cols)
{
	auto merged = buildMergedList(state.processLists, state.host);
	sortMergedList(merged, sortBy);

	const ProcTableLayout table = procTableLayout(cols);
	const auto hdr = [&](std::string_view label, SortBy key) -> std::string {
		return (sortBy == key) ? xpum::compat::format("{}*", label) : std::string{label};
	};
	const std::string gap(PROC_GAP, ' ');

	// One function lays out the header and every row, so the columns cannot drift apart.
	const auto line = [&](std::string_view pid, std::string_view gpu, std::string_view type, std::string_view gpuMem,
						  std::string_view gpuMemPct, std::string_view cmp, std::string_view rnd, std::string_view med,
						  std::string_view cpy, std::string_view cpu, std::string_view hostMem, std::string_view name) {
		std::string out = xpum::compat::format("{:{}}{:>{}}{}{:>{}}{}{:<{}}{}{:>{}}{}", "", PROC_INDENT, pid, COL_PID,
											   gap, gpu, COL_GPU, gap, type, COL_TYPE, gap, gpuMem, COL_GPU_MEM, gap);
		if (table.showGpuMemPercent) {
			out += xpum::compat::format("{:>{}}{}", gpuMemPct, COL_PERCENT, gap);
		}
		out += xpum::compat::format("{:>{}}{}", cmp, COL_PERCENT, gap);
		if (table.showRender) {
			out += xpum::compat::format("{:>{}}{}", rnd, COL_PERCENT, gap);
		}
		out += xpum::compat::format("{:>{}}{}", med, COL_PERCENT, gap);
		if (table.showCopy) {
			out += xpum::compat::format("{:>{}}{}", cpy, COL_PERCENT, gap);
		}
		const std::string_view shown = name.substr(0, static_cast<std::size_t>(table.nameWidth));
		out += xpum::compat::format("{:>{}}{}{:>{}}{}{:<{}}", cpu, COL_PERCENT, gap, hostMem, COL_HOST_MEM, gap, shown,
									table.nameWidth);
		if (shown.size() < name.size() && !shown.empty()) {
			out.back() = '~';
		}
		return out;
	};

	// Header row - always bold + underline
	term::gotoPos(startRow, 0);
	term::attrBold();
	term::attrUL();
	std::fputs(line("PID", "GPU", "Type", hdr("GPUMiB", SortBy::Memory), "GMem%", hdr("Cmp%", SortBy::Compute), "Rnd%",
					"Med%", "Cpy%", hdr("CPU%", SortBy::Cpu), "HostMiB", "Name")
				   .c_str(),
			   stdout);
	term::attrReset();
	term::clearToEol();

	const int total = static_cast<int>(merged.size());
	const int maxVisible = std::max(0, std::min(total - state.scrollOffset, procH));

	const auto pct = [](std::optional<float> v) -> std::string {
		return v ? xpum::compat::format("{:.1f}", *v) : std::string{"N/A"};
	};

	for (int i = 0; i < maxVisible; ++i) {
		const int absIdx = state.scrollOffset + i;
		const auto &mp = merged[static_cast<std::size_t>(absIdx)];
		const psInfo *p = mp.proc;
		const bool sel = (absIdx == state.selectedRow);

		term::gotoPos(startRow + 1 + i, 0);
		if (sel) {
			if (g_colorsEnabled) {
				term::setFg(term::FG_BLACK);
				term::setBg(term::BG_WHITE);
			} else {
				term::attrReverse();
			}
		}
		const std::string hostMem =
			mp.host.resident ? std::to_string(mp.host.resident->mebibytes()) : std::string{"N/A"};
		std::fputs(line(std::to_string(p->processId), std::to_string(mp.gpuIdx), engineType(p->engines),
						std::to_string(p->memSize / 1024), pct(gpuMemPercentOf(state, mp)), pct(p->util.compute),
						pct(p->util.render), pct(p->util.media), pct(p->util.copy), pct(mp.host.cpu), hostMem,
						p->commandName)
					   .c_str(),
				   stdout);
		term::attrReset();
		term::clearToEol();
	}

	// Rows a shorter list no longer fills would otherwise keep last frame's text.
	for (int i = maxVisible; i < procH; ++i) {
		term::gotoPos(startRow + 1 + i, 0);
		term::clearToEol();
	}

	if (total > procH) {
		term::gotoPos(startRow + 1 + maxVisible, 0);
		if (g_colorsEnabled) {
			term::attrDim();
		}
		std::fprintf(stdout, "  ... showing %d-%d of %d  (j/k scroll  g/G top/bottom)", state.scrollOffset + 1,
					 state.scrollOffset + maxVisible, total);
		if (g_colorsEnabled) {
			term::attrReset();
		}
		term::clearToEol();
		term::attrReset();
	}
}

void renderHeader(int cols, bool euHeld)
{
	const std::string title =
		xpum::compat::format("  xpu-smi top  {}", formatLocalTime(std::chrono::system_clock::now()));
	constexpr std::string_view euNotice = "  EU: needs root or CAP_PERFMON";

	term::gotoPos(0, 0);
	term::attrBold();
	if (g_colorsEnabled) {
		term::setFg(term::FG_WHITE);
	}
	std::fputs(title.c_str(), stdout);
	term::attrReset();
	std::size_t used = title.size();
	if (!euHeld) {
		if (g_colorsEnabled) {
			term::attrDim();
		}
		std::fputs(euNotice.data(), stdout);
		term::attrReset();
		used += euNotice.size();
	}
	const int pad = cols - static_cast<int>(used);
	if (pad > 0) {
		std::fprintf(stdout, "%*s", pad, "");
	}
}

/// Asks which signal to send to @p proc; nullopt when the user cancels.
std::optional<StopSignal> confirmKill(const psInfo &proc, const TerminalSize &screen)
{
	const std::string msg = xpum::compat::format("  Kill {} ({}) ?  [ t=SIGTERM   k=SIGKILL   N=cancel ]  ",
												 proc.processId, proc.commandName);

	const int col = std::max(0, (static_cast<int>(screen.columns) - static_cast<int>(msg.size())) / 2);
	term::gotoPos(static_cast<int>(screen.rows) / 2, col);
	term::attrReverse();
	term::attrBold();
	std::fputs(msg.c_str(), stdout);
	term::attrReset();
	term::flushOut();

	const int ch = keyboard::readKey(std::nullopt); // block until keypress

	if (ch == 't' || ch == 'T') {
		return StopSignal::Terminate;
	}
	if (ch == 'k' || ch == 'K') {
		return StopSignal::Kill;
	}
	return std::nullopt;
}

void recordKill(TopState &state, uint32_t pid, StopSignal sig, SignalResult result)
{
	state.statusFailed = (result != SignalResult::Sent);
	switch (result) {
	case SignalResult::Sent:
		state.status = xpum::compat::format("sent {} to {}", sig == StopSignal::Kill ? "SIGKILL" : "SIGTERM", pid);
		break;
	case SignalResult::NotPermitted:
		state.status = xpum::compat::format("{}: not permitted (another user's process?)", pid);
		break;
	case SignalResult::NoSuchProcess:
		state.status = xpum::compat::format("{}: already exited", pid);
		break;
	case SignalResult::Failed:
		state.status = xpum::compat::format("{}: signal failed", pid);
		break;
	}
}

/// Gathers one device's processes and their engine use over a window, from the device's
/// snapshots of the walks that opened and closed it, and smooths the result.
void refreshProcesses(const devInfo &dev, std::vector<psInfo> &procs,
					  const std::vector<fdinfo::ProcessSnapshot> &before,
					  const std::vector<fdinfo::ProcessSnapshot> &after, const ProcessFacts &facts,
					  std::unordered_map<uint32_t, SmoothedUtil> &smoothed)
{
	procs.clear();
	// The process list comes from the same snapshots, so Level Zero walks /proc for none of
	// it; it is still asked when fdinfo cannot tell (no DRM fdinfo on this kernel).
	const auto known = cmdPs::processStatesFromFdinfo(before, after, getCurrentProcessId());
	std::optional<uint64_t> selfMemKiB;
	if (const ze_result_t r = cmdPs::getProcessList(&dev, procs, &selfMemKiB, known ? &*known : nullptr);
		isFatalPsError(r)) {
		DBG("top: process list for device {} failed: 0x{:X} ({})\n", dev.index, static_cast<unsigned>(r),
			l0_error_to_string(r));
	}
	cmdPs::dropIdleRuntimeContexts(procs, 0, before, after, selfMemKiB, facts);

	const auto utilMap = fdinfo::delta(before, after);
	std::unordered_map<uint32_t, const fdinfo::ProcessSnapshot *> snapIdx;
	for (const auto &snap : after) {
		snapIdx[snap.pid] = &snap;
	}
	for (auto &p : procs) {
		if (p.engines == 0) {
			if (const auto sit = snapIdx.find(p.processId); sit != snapIdx.end()) {
				p.engines = fdinfo::enginesFromSnapshot(*sit->second);
			}
		}
		if (const auto it = utilMap.find(p.processId); it != utilMap.end()) {
			p.util = fdinfo::toProcUtil(it->second);
		}
	}

	smoothProcessList(procs, smoothed);
}

/// Reads CPU time and resident memory for every listed pid; forgets pids no longer listed.
void refreshHostUsage(TopState &state)
{
	std::unordered_map<uint32_t, HostReading> readings;
	HostUsageMap host;
	for (const auto &list : state.processLists) {
		for (const auto &p : list) {
			if (readings.contains(p.processId)) {
				continue;
			}
			const auto sample = readProcessHostSample(p.processId);
			if (!sample) {
				host[p.processId] = {};
				continue;
			}
			const HostReading now{.cpuTime = sample->cpuTime, .at = std::chrono::steady_clock::now()};
			HostUsage usage{.cpu = std::nullopt, .resident = sample->resident};
			if (const auto prev = state.hostReadings.find(p.processId); prev != state.hostReadings.end()) {
				usage.cpu = cpuPercent(prev->second, now);
			}
			readings[p.processId] = now;
			host[p.processId] = usage;
		}
	}
	state.hostReadings = std::move(readings);
	state.host = std::move(host);
}

/// Every device's processes over the window from @p before to @p after.
void refreshAllProcesses(TopState &state, std::span<devInfo> devices, const fdinfo::Walk &before,
						 const fdinfo::Walk &after, const std::unordered_set<std::string> &gpuPciAddrs)
{
	// Built per tick: a process can open another GPU later, and a pid can be reused.
	const ProcessFacts facts{gpuPciAddrs};
	for (std::size_t i = 0; i < devices.size(); ++i) {
		const std::string pci = devPciAddr(devices[i]);
		refreshProcesses(devices[i], state.processLists[i], fdinfo::snapshotsFor(before.devices, pci),
						 fdinfo::snapshotsFor(after.devices, pci), facts, state.smoothed[i]);
	}
	refreshHostUsage(state);
}

// Plain frames, one after another: the in-place redraw of TableBuilder's streaming mode would put
// cursor movement into a pipe or a log.
void renderBatch(const TopState &state, std::span<devInfo> devices, SortBy sortBy)
{
	auto devTable = TableBuilder::bordered();
	devTable.addColumn("GPU", Align::Right)
		.addColumn("GPU%", Align::Right)
		.addColumn("Cmp%", Align::Right)
		.addColumn("MemUsed", Align::Right)
		.addColumn("Temp", Align::Right)
		.addColumn("Power", Align::Right)
		.addColumn("Limit", Align::Right)
		.addColumn("Fan", Align::Right)
		.addColumn("Freq", Align::Right);

	auto procTable = TableBuilder::bordered();
	procTable.addColumn("PID", Align::Right)
		.addColumn("GPU", Align::Right)
		.addColumn("Name", Align::Left)
		.addColumn("GPUMiB", Align::Right)
		.addColumn("GMem%", Align::Right)
		.addColumn("Cmp%", Align::Right)
		.addColumn("Med%", Align::Right)
		.addColumn("Cpy%", Align::Right)
		.addColumn("CPU%", Align::Right)
		.addColumn("HostMiB", Align::Right);

	for (std::size_t i = 0; i < devices.size(); ++i) {
		auto &dev = devices[i];
		const auto &cache = state.caches[i];
		const auto cell = [&](std::string_view f) { return orNotAvailable(display(dev, f, cache)); };
		devTable.addRow(std::to_string(dev.index), cell("utilization.gpu"), cell("utilization.compute"),
						cell("memory.used"), cell("temperature.gpu"), cell("power.draw"), cell("power.limit"),
						cell("fan.speed"), cell("clocks.current.graphics"));
	}

	auto merged = buildMergedList(state.processLists, state.host);
	sortMergedList(merged, sortBy);
	const auto fmtU = [](std::optional<float> v) -> std::string {
		return v ? xpum::compat::format("{:.1f}%", *v) : "N/A";
	};
	for (const auto &mp : merged) {
		const psInfo *p = mp.proc;
		procTable.addRow(std::to_string(p->processId), std::to_string(mp.gpuIdx), p->commandName,
						 std::to_string(p->memSize / 1024), fmtU(gpuMemPercentOf(state, mp)), fmtU(p->util.compute),
						 fmtU(p->util.media), fmtU(p->util.copy), fmtU(mp.host.cpu),
						 mp.host.resident ? std::to_string(mp.host.resident->mebibytes()) : std::string{"N/A"});
	}

	devTable.print();
	procTable.print();
	std::putchar('\n');
	std::fflush(stdout);
}

void renderFrame(TopState &state, std::span<devInfo> devices, SortBy sortBy, bool euHeld)
{
	const int rows = static_cast<int>(state.screen.rows);
	const int cols = static_cast<int>(state.screen.columns);

	term::home();
	renderHeader(cols, euHeld);

	int nextRow = 1;
	for (std::size_t i = 0; i < devices.size(); ++i) {
		nextRow = renderDevicePanel(nextRow, devices[i], state.caches[i], state.history[i], state.layout);
	}
	state.layout.procTop = nextRow;
	state.layout.procH = std::max(1, rows - nextRow - 2);

	renderProcessTable(state, sortBy, state.layout.procTop, state.layout.procH, cols);

	const int total = static_cast<int>(buildMergedList(state.processLists, state.host).size());
	term::gotoPos(rows - 1, 0);
	if (g_colorsEnabled) {
		term::attrDim();
	}
	std::fprintf(stdout, "  q:quit  j/k:scroll  K:kill  g/G:top/bottom  %d/%d", total > 0 ? state.selectedRow + 1 : 0,
				 total);
	if (g_colorsEnabled) {
		term::attrReset();
	}
	if (!state.status.empty()) {
		if (g_colorsEnabled && state.statusFailed) {
			term::setFg(term::FG_RED);
		}
		std::fprintf(stdout, "   %s", state.status.c_str());
		term::attrReset();
	}
	term::clearToEol();
	term::clearToEnd();
	term::flushOut();
}

bool handleInput(int ch, TopState &state, SortBy sortBy)
{
	if (ch == keyboard::NO_KEY) {
		return false;
	}
	state.status.clear();

	auto merged = buildMergedList(state.processLists, state.host);
	sortMergedList(merged, sortBy);
	const int total = static_cast<int>(merged.size());
	const int maxSel = std::max(0, total - 1);
	const int maxScroll = std::max(0, total - state.layout.procH);

	switch (ch) {
	case 'q':
	case 'Q':
	case keyboard::KEY_ESCAPE:
	case CTRL_C:
		g_quit.store(true, std::memory_order_relaxed);
		return true;

	case 'j':
	case keyboard::KEY_DOWN:
		state.selectedRow = std::min(state.selectedRow + 1, maxSel);
		if (state.selectedRow >= state.scrollOffset + state.layout.procH) {
			++state.scrollOffset;
		}
		break;

	case 'k':
	case keyboard::KEY_UP:
		state.selectedRow = std::max(state.selectedRow - 1, 0);
		if (state.selectedRow < state.scrollOffset) {
			--state.scrollOffset;
		}
		break;

	case keyboard::KEY_PAGE_DOWN:
	case 'f': {
		const int step = state.layout.procH;
		state.selectedRow = std::min(state.selectedRow + step, maxSel);
		state.scrollOffset = std::min(state.scrollOffset + step, maxScroll);
		break;
	}
	case keyboard::KEY_PAGE_UP:
	case 'b': {
		const int step = state.layout.procH;
		state.selectedRow = std::max(state.selectedRow - step, 0);
		state.scrollOffset = std::max(state.scrollOffset - step, 0);
		break;
	}
	case 'g':
		state.selectedRow = 0;
		state.scrollOffset = 0;
		break;

	case 'G':
		state.selectedRow = maxSel;
		state.scrollOffset = maxScroll;
		break;

	case 'K':
		if (state.selectedRow < total) {
			const psInfo *target = merged[static_cast<std::size_t>(state.selectedRow)].proc;
			if (const auto sig = confirmKill(*target, state.screen); sig) {
				recordKill(state, target->processId, *sig, sendStopSignal(target->processId, *sig));
			}
		}
		break;

	default:
		break;
	}
	return false;
}

} // anonymous namespace

/**
 * @brief top is sysman-only unless the EU bars are available.
 *
 * @param[in] cmdArgs  argv tokens after the subcommand name (unused).
 *
 * @retval {.driver = DriverMode::SysmanOnly, .optional = EU bars}  Always.
 */
CommandPlan cmdTop::needs(UNUSED std::span<char *const> cmdArgs) const { return {.optional = TOP_FEATURES}; }

void cmdTop::help(HELP helpType)
{
	std::vector<helpCmd> helpList;
	helpList.push_back(helpCmd(TITLE, "Live GPU process monitor"));
	helpList.push_back(helpCmd(BLANK));
	helpList.push_back(helpCmd(TITLE, "Usage: %s top [Options]", progName.c_str()));
	helpList.push_back(helpCmd(BLANK));
	helpList.push_back(helpCmd(HEADING, "--interval <sec>       Refresh interval in seconds (default 1, min 0.1)"));
	helpList.push_back(helpCmd(HEADING, "--count <N>            Exit after N frames (0 = unlimited, default)"));
	helpList.push_back(helpCmd(HEADING, "--device <idx|BDF>     Limit to one GPU (default: all)"));
	helpList.push_back(
		helpCmd(HEADING, "--sort <key>           Process sort: mem (default, GPU memory), compute, cpu, pid"));
	helpList.push_back(
		helpCmd(HEADING, "--no-color             Disable ANSI colour (useful for TERM=dumb or logging)"));
	helpList.push_back(helpCmd(HEADING, "--batch                Plain table output without the TUI (for scripts)"));
	printHelp(helpList, helpType);
}

int cmdTop::run(arg_struct *args)
{
	double intervalSeconds = 1.0;
	int count = 0;
	bool batch = false;
	bool noColor = false;
	std::string sortStr = "mem";
	std::string deviceSpec;

	CLI::App sub{"Live GPU monitor", "top"};
	sub.set_help_flag("-h,--help", "Print this help message and exit");
	sub.add_option("--interval", intervalSeconds, "Refresh interval in seconds (default 1, min 0.1)")
		->check(CLI::Range(0.1, 3600.0));
	sub.add_option("--count", count, "Exit after N frames (0 = unlimited)")->check(CLI::NonNegativeNumber);
	sub.add_flag("--batch", batch, "Plain table output without the TUI");
	sub.add_flag("--no-color", noColor, "Disable ANSI colour output");
	sub.add_option("--sort", sortStr, "Process sort: mem (default), compute, cpu, pid");
	sub.add_option("--device", deviceSpec, "GPU index or BDF (default: all)");

	try {
		sub.parse(args->argc - 1, args->argv + 1);
	} catch (const CLI::CallForHelp &) {
		help();
		return 0;
	} catch (const CLI::ParseError &e) {
		ERR("{}", e.what());
		ERR("Run with --help for more information.\n");
		return 1;
	}

	const auto sortKey = parseSortKey(sortStr);
	if (!sortKey) {
		ERR("top: --sort takes mem, compute, cpu or pid\n");
		return 1;
	}
	const SortBy sortBy = *sortKey;

	const auto interval =
		std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>{intervalSeconds});

	std::vector<devInfo> deviceList;
	if (args->sm.findDevice(deviceSpec.c_str(), &deviceList) != ZE_RESULT_SUCCESS || deviceList.empty()) {
		ERR("top: no GPU devices found\n");
		return 1;
	}
	const std::size_t numDevices = deviceList.size();

	// A core device handle exists only when the EU feature was admitted.
	const bool euHeld = std::ranges::any_of(deviceList, [](const devInfo &d) { return d.deviceHdl != nullptr; });
	const metrics::MetricInput inputs = topInputs(euHeld);

	TopState state;
	state.caches.resize(numDevices);
	state.history.resize(numDevices);
	state.processLists.resize(numDevices);
	state.smoothed.resize(numDevices);
	for (auto &dev : deviceList) {
		const auto total = queryField(dev, "memory.total", metrics::MetricCache{});
		state.gpuMemTotalMiB.push_back(total ? parseNumber(total->raw) : std::nullopt);
	}

	g_quit.store(false, std::memory_order_relaxed);
	const StopSignals stopSignals;

	// Destroyed in reverse: the screen is restored before the terminal modes.
	const bool tui = !batch && stdoutIsTerminal();
	std::optional<keyboard::RawKeyboard> rawKeys;
	std::optional<term::AltScreen> altScreen;
	if (tui) {
		rawKeys.emplace();
		altScreen.emplace();
		g_colorsEnabled = shouldUseColor(noColor);
		state.screen = stdoutSize().value_or(state.screen);
	} else {
		g_colorsEnabled = false;
	}
	const bool haveKeys = rawKeys && rawKeys->active();

	const auto euShown = [&] { return std::ranges::any_of(state.caches, [](const auto &c) { return c.euAvail; }); };
	const auto layoutNow = [&] {
		return calcLayout(static_cast<int>(state.screen.rows), static_cast<int>(state.screen.columns),
						  static_cast<int>(numDevices), euShown());
	};

	// Every tick walks /proc in full: the cheaper walkSince() skips processes that held no DRM
	// client in the walk before, so chained from tick to tick it would never see a running
	// process that opens a GPU later.
	const std::unordered_set<std::string> gpuPciAddrs = devPciAddrs(deviceList);
	const auto walkNow = [&] { return metrics::captureFdinfo(deviceList, inputs | metrics::MetricInput::FDINFO); };

	// First sample window: begin before-snapshots, sleep, end after-snapshots
	const auto firstDeadline = std::chrono::steady_clock::now() + interval;
	const fdinfo::Walk firstWalk = walkNow();
	for (std::size_t i = 0; i < numDevices; ++i) {
		state.caches[i] = metrics::populateMetricCacheBegin(deviceList[i], inputs, &firstWalk.devices);
	}
	std::this_thread::sleep_until(firstDeadline);
	state.lastWalk = walkNow();
	for (std::size_t i = 0; i < numDevices; ++i) {
		metrics::populateMetricCacheEnd(deviceList[i], state.caches[i], &state.lastWalk.devices);
	}

	// EU availability is known only now
	if (tui) {
		state.layout = layoutNow();
	}

	refreshAllProcesses(state, deviceList, firstWalk, state.lastWalk, gpuPciAddrs);

	int remaining = count;

	while (!g_quit.load(std::memory_order_relaxed)) {
		if (tui) {
			const TerminalSize now = stdoutSize().value_or(state.screen);
			if (now != state.screen) {
				state.screen = now;
				state.layout = layoutNow();
			}
			renderFrame(state, deviceList, sortBy, euHeld);
		} else {
			renderBatch(state, deviceList, sortBy);
		}

		if (remaining > 0 && --remaining == 0) {
			break;
		}

		const auto tickDeadline = std::chrono::steady_clock::now() + interval;
		while (std::chrono::steady_clock::now() < tickDeadline && !g_quit.load(std::memory_order_relaxed)) {
			if (haveKeys) {
				if (handleInput(keyboard::readKey(KEY_POLL), state, sortBy)) {
					break;
				}
			} else {
				std::this_thread::sleep_for(KEY_POLL);
			}
		}

		if (g_quit.load(std::memory_order_relaxed)) {
			break;
		}

		fdinfo::Walk walk = walkNow();
		for (std::size_t i = 0; i < numDevices; ++i) {
			state.caches[i] =
				metrics::populateMetricCacheContinuous(deviceList[i], state.caches[i], inputs, &walk.devices);
		}
		refreshAllProcesses(state, deviceList, state.lastWalk, walk, gpuPciAddrs);
		state.lastWalk = std::move(walk);
	}

	return 0;
}
