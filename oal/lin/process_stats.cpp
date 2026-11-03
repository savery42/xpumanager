/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "process_stats.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <string_view>
#include <unistd.h>

namespace {

// Fields of /proc/<pid>/stat, numbered from 1 as in proc(5).
constexpr std::size_t FIRST_FIELD_AFTER_COMM = 3; // state
constexpr std::size_t FIELD_UTIME = 14;
constexpr std::size_t FIELD_STIME = 15;
constexpr std::size_t FIELD_RSS = 24; // in pages

[[nodiscard]] std::optional<uint64_t> parseUnsigned(std::string_view s) noexcept
{
	uint64_t v = 0;
	const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), v);
	if (ec != std::errc{} || ptr != s.data() + s.size()) {
		return std::nullopt;
	}
	return v;
}

} // namespace

std::optional<ProcessHostSample> readProcessHostSample(std::uint32_t pid, const std::string &procRoot)
{
	std::ifstream in{procRoot + "/" + std::to_string(pid) + "/stat"};
	if (!in) {
		return std::nullopt;
	}
	const std::string line{std::istreambuf_iterator<char>{in}, std::istreambuf_iterator<char>{}};

	// The command name (field 2) is in parentheses and may itself contain spaces and ')', so the
	// fields that follow start after the last ')'.
	const auto close = line.rfind(')');
	if (close == std::string::npos) {
		return std::nullopt;
	}
	std::array<std::string_view, FIELD_RSS - FIRST_FIELD_AFTER_COMM + 1> fields{};
	std::string_view rest = std::string_view{line}.substr(close + 1);
	for (auto &field : fields) {
		const auto start = rest.find_first_not_of(" \n");
		if (start == std::string_view::npos) {
			return std::nullopt;
		}
		rest.remove_prefix(start);
		const auto end = rest.find_first_of(" \n");
		field = rest.substr(0, end);
		rest.remove_prefix(end == std::string_view::npos ? rest.size() : end);
	}
	const auto field = [&](std::size_t n) { return parseUnsigned(fields[n - FIRST_FIELD_AFTER_COMM]); };
	const auto utime = field(FIELD_UTIME);
	const auto stime = field(FIELD_STIME);
	const auto rssPages = field(FIELD_RSS);
	const long ticksPerSecond = sysconf(_SC_CLK_TCK);
	const long pageSize = sysconf(_SC_PAGESIZE);
	if (!utime || !stime || !rssPages || ticksPerSecond <= 0 || pageSize <= 0) {
		return std::nullopt;
	}

	const auto ticks = *utime + *stime;
	const auto hz = static_cast<uint64_t>(ticksPerSecond);
	const std::chrono::nanoseconds cpuTime{
		static_cast<std::chrono::nanoseconds::rep>(ticks / hz * 1'000'000'000ULL + ticks % hz * 1'000'000'000ULL / hz)};
	return ProcessHostSample{.cpuTime = cpuTime,
							 .resident = xpum::units::Bytes{*rssPages * static_cast<uint64_t>(pageSize)}};
}
