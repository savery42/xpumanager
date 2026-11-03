/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "local_time.h"

#include <array>
#include <ctime>

std::string formatLocalTime(std::chrono::system_clock::time_point when)
{
	const std::time_t seconds = std::chrono::system_clock::to_time_t(std::chrono::floor<std::chrono::seconds>(when));
	std::tm local{};
	tzset(); // localtime_r need not re-read TZ
	if (localtime_r(&seconds, &local) == nullptr) {
		return {};
	}

	std::array<char, 64> buf{};
	if (std::strftime(buf.data(), buf.size(), "%H:%M:%S %Z", &local) == 0) {
		return {};
	}
	std::string text{buf.data()};
	if (text.ends_with(' ')) {
		// Empty abbreviation: fall back to the numeric offset.
		if (std::strftime(buf.data(), buf.size(), "%H:%M:%S %z", &local) == 0) {
			return {};
		}
		text = buf.data();
	}
	return text;
}
