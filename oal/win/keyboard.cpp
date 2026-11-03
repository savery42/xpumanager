/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "keyboard.h"

#include <thread>

struct keyboard::RawKeyboard::Saved
{};

keyboard::RawKeyboard::RawKeyboard() = default;

keyboard::RawKeyboard::~RawKeyboard() = default;

int keyboard::readKey(std::optional<std::chrono::milliseconds> timeout) noexcept
{
	if (timeout) {
		std::this_thread::sleep_for(*timeout);
	}
	return NO_KEY;
}
