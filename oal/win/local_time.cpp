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
	if (localtime_s(&local, &seconds) != 0) {
		return {};
	}

	// The Windows C runtime spells %Z out in full ("Pacific Standard Time").
	std::array<char, 96> buf{};
	if (std::strftime(buf.data(), buf.size(), "%H:%M:%S %Z", &local) == 0) {
		return {};
	}
	std::string text{buf.data()};
	if (text.ends_with(' ')) {
		if (std::strftime(buf.data(), buf.size(), "%H:%M:%S %z", &local) == 0) {
			return {};
		}
		text = buf.data();
	}
	return text;
}
