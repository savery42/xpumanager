/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 *
 * Tests for formatLocalTime(). TZ is set to POSIX rule strings, so no zoneinfo files are needed.
 */

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "local_time.h"

#include <chrono>
#include <cstdlib>

namespace {

using namespace std::chrono_literals;

std::chrono::system_clock::time_point utc(int month, int day, std::chrono::hours hour)
{
	return std::chrono::sys_days{std::chrono::year{2026} / month / day} + hour;
}

void setZone(const char *tz) { REQUIRE(setenv("TZ", tz, 1) == 0); }

} // namespace

TEST_CASE("formatLocalTime shows local time with the zone abbreviation, standard and daylight")
{
	setZone("PST8PDT,M3.2.0,M11.1.0");
	CHECK(formatLocalTime(utc(1, 15, 12h)) == "04:00:00 PST");
	CHECK(formatLocalTime(utc(7, 15, 12h)) == "05:00:00 PDT");
}

TEST_CASE("formatLocalTime handles UTC and zones whose abbreviation is an offset")
{
	setZone("UTC0");
	CHECK(formatLocalTime(utc(1, 15, 12h)) == "12:00:00 UTC");

	setZone("<+03>-3");
	CHECK(formatLocalTime(utc(1, 15, 12h)) == "15:00:00 +03");
}

TEST_CASE("formatLocalTime truncates to whole seconds")
{
	setZone("UTC0");
	CHECK(formatLocalTime(utc(1, 15, 12h) + 999ms) == "12:00:00 UTC");
}
