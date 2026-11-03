/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Tests for readProcessHostSample() against fixture stat files, and once against this process.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "process_stats.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

// Writes <root>/<pid>/stat and removes the tree when it goes out of scope.
class FakeProc
{
public:
	FakeProc() : root(fs::temp_directory_path() / ("xpum_process_stats_" + std::to_string(::getpid()))) {}
	~FakeProc() { fs::remove_all(root); }
	FakeProc(const FakeProc &) = delete;
	FakeProc &operator=(const FakeProc &) = delete;
	FakeProc(FakeProc &&) = delete;
	FakeProc &operator=(FakeProc &&) = delete;

	void write(uint32_t pid, const std::string &stat) const
	{
		fs::create_directories(root / std::to_string(pid));
		std::ofstream{root / std::to_string(pid) / "stat"} << stat;
	}
	[[nodiscard]] std::string path() const { return root.string(); }

private:
	fs::path root;
};

// A stat line with utime, stime and rss set and every other field zero. comm is given unbracketed.
std::string statLine(uint32_t pid, const std::string &comm, uint64_t utime, uint64_t stime, uint64_t rssPages)
{
	std::string s = std::to_string(pid) + " (" + comm + ") S";
	for (int field = 4; field <= 52; ++field) {
		s += ' ';
		if (field == 14) {
			s += std::to_string(utime);
		} else if (field == 15) {
			s += std::to_string(stime);
		} else if (field == 24) {
			s += std::to_string(rssPages);
		} else {
			s += '0';
		}
	}
	return s + '\n';
}

const auto TICK = std::chrono::nanoseconds{1'000'000'000 / sysconf(_SC_CLK_TCK)};
const auto PAGE = static_cast<uint64_t>(sysconf(_SC_PAGESIZE));

} // namespace

TEST_CASE("readProcessHostSample sums user and system time and converts pages to bytes")
{
	const FakeProc proc;
	proc.write(42, statLine(42, "gpu_stress", 300, 20, 1000));
	const auto sample = readProcessHostSample(42, proc.path());
	REQUIRE(sample.has_value());
	CHECK(sample->cpuTime == 320 * TICK);
	CHECK(sample->resident.bytes() == 1000 * PAGE);
}

TEST_CASE("readProcessHostSample parses a command name holding spaces and parentheses")
{
	const FakeProc proc;
	proc.write(7, statLine(7, "a b) (c", 5, 5, 3));
	const auto sample = readProcessHostSample(7, proc.path());
	REQUIRE(sample.has_value());
	CHECK(sample->cpuTime == 10 * TICK);
	CHECK(sample->resident.bytes() == 3 * PAGE);
}

TEST_CASE("readProcessHostSample rejects what it cannot read")
{
	const FakeProc proc;
	CHECK_FALSE(readProcessHostSample(99, proc.path()).has_value());
	proc.write(8, "8 (short) S 1 2 3\n");
	CHECK_FALSE(readProcessHostSample(8, proc.path()).has_value());
	proc.write(9, "9 no parens\n");
	CHECK_FALSE(readProcessHostSample(9, proc.path()).has_value());
}

TEST_CASE("readProcessHostSample reads this process")
{
	const auto sample = readProcessHostSample(static_cast<uint32_t>(::getpid()));
	REQUIRE(sample.has_value());
	CHECK(sample->resident.bytes() > 0);
}
