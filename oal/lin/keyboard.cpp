/*
 * Copyright (C) 2026 Intel Corporation
 * SPDX-License-Identifier: MIT
 */

#include "keyboard.h"

#include <array>
#include <cstddef>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

struct keyboard::RawKeyboard::Saved
{
	termios state{};
};

keyboard::RawKeyboard::RawKeyboard()
{
	if (isatty(STDIN_FILENO) == 0) {
		return;
	}
	termios original{};
	if (tcgetattr(STDIN_FILENO, &original) != 0) {
		return;
	}
	termios raw = original;
	cfmakeraw(&raw);
	raw.c_cc[VMIN] = 0;
	raw.c_cc[VTIME] = 0;
	if (tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) != 0) {
		return;
	}
	saved = std::make_unique<Saved>(Saved{original});
}

keyboard::RawKeyboard::~RawKeyboard()
{
	if (saved) {
		tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved->state);
	}
}

int keyboard::readKey(std::optional<std::chrono::milliseconds> timeout) noexcept
{
	pollfd fd{STDIN_FILENO, POLLIN, 0};
	const int waitMs = timeout ? static_cast<int>(timeout->count()) : -1;
	if (poll(&fd, 1, waitMs) <= 0) {
		return NO_KEY;
	}
	std::array<unsigned char, 8> buf{};
	const ssize_t n = read(STDIN_FILENO, buf.data(), buf.size());
	if (n <= 0) {
		return NO_KEY;
	}
	return decodeKey(std::span{buf}.first(static_cast<std::size_t>(n)));
}
